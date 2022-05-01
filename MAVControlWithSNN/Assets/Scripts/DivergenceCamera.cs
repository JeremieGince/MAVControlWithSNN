using System;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Rendering;
using cv2 = OpenCvSharp;



public class DivergenceSensor: ISensor, IDisposable {

    private Camera m_CameraInput;
    private Texture2D m_TextureInput;
    private string m_SensorName;

    private cv2.Mat m_PrevImgGray;
    private cv2.Point2f[] m_PrevKeypoints;
    private cv2.Mat m_CurrImgGray;
    private cv2.Point2f[] m_CurrKeypoints;
    private float m_PrevRenderTime = 0f;
    private float m_CurrRanderTime = 0f;
    private float m_Divergence = 0f;

    private int m_MaxDivergencePoints;
    private bool m_DivergenceAsOneHot;
    private int m_DivergenceBins = 20;
    private float m_DivergenceBinSize = 10f;

    private int m_FastThreshold;
    private bool m_IsStacked = false;
    private bool m_IsReady = false;

    public DivergenceSensor(Camera p_CameraInput, string p_SensorName, int p_FastThreshold, int p_MaxDivergencePoints, bool p_DivergenceAsOneHot, int p_DivergenceBins = 20, float p_DivergenceBinSize = 10f, bool p_IsStacked = false) {
        m_CameraInput = p_CameraInput;
        m_SensorName = p_SensorName;
        m_FastThreshold = p_FastThreshold;

        m_MaxDivergencePoints = p_MaxDivergencePoints;
        m_DivergenceAsOneHot = p_DivergenceAsOneHot;
        m_DivergenceBins = p_DivergenceBins;
        m_DivergenceBinSize = p_DivergenceBinSize;

        m_IsStacked = p_IsStacked;

        m_TextureInput = new Texture2D(m_CameraInput.pixelWidth, m_CameraInput.pixelHeight, TextureFormat.RGB24, mipChain: false);
        m_IsReady = false;
    }

    byte[] ISensor.GetCompressedObservation() {
        return Array.ConvertAll(GetObservationList(), new Converter<float, byte>(Convert.ToByte));
    }

    CompressionSpec ISensor.GetCompressionSpec() {
        return new CompressionSpec(SensorCompressionType.None);
    }

    string ISensor.GetName() {
        return m_SensorName;
    }

    ObservationSpec ISensor.GetObservationSpec() {
        if (m_IsStacked) return ObservationSpec.Visual(GetOutputSize(), 1, 1);
        return ObservationSpec.Vector(GetOutputSize());
    }

    void ISensor.Reset() {
        m_Divergence = 0f;
    }

    void ISensor.Update() {
        UpdateState();
    }

    int ISensor.Write(ObservationWriter writer) {
        float[] obs = GetObservationList();
        writer.AddList(obs);
        return obs.Length;
    }

    public void Dispose() {
        m_PrevImgGray.Dispose();
        m_CurrImgGray.Dispose();

        if ((object)m_TextureInput != null) {
            DestroyTexture(m_TextureInput);
            m_TextureInput = null;
        }
    }

    public float UpdateState() {
        UpdateCurrentFrame();
        if (m_IsReady) {
            bool valid = ComputeCurrentKeypoints();
            if (valid) UpdateDivergence();
        }
        UpdatePreviousKeypoints();
        return m_Divergence;
    }

    public static void DestroyTexture(Texture2D texture) {
        if (Application.isEditor) {
            UnityEngine.Object.DestroyImmediate(texture);
        }
        else {
            UnityEngine.Object.Destroy(texture);
        }
    }


    /// <summary>
    /// Update the current states of the camera. 
    ///    -> Render the camera manualy.
    ///    -> update the current render time.
    ///    -> update currImgGray that contain the current frame in grayscale color.
    /// </summary>
    private void UpdateCurrentFrame() {
        m_CurrRanderTime = Time.time * Time.timeScale;
        RawInputToTexture(m_CameraInput, m_TextureInput);

        cv2.Mat img = cv2.Unity.TextureToMat(m_TextureInput);
        m_CurrImgGray = new cv2.Mat(rows: img.Rows, cols: img.Cols, type: cv2.MatType.CV_32F);
        cv2.Cv2.CvtColor(img, m_CurrImgGray, cv2.ColorConversionCodes.BGR2GRAY);
    }

    /// <summary>
    /// The current states go to the previous states.
    ///     -> previous frame become the current frame
    ///     -> previous render time become the current render time.
    ///     -> the previous keypoints are compute with the FASt algorithm from the current frame
    ///     -> set the isReady flag to true
    /// </summary>
    private void UpdatePreviousKeypoints() {
        m_PrevImgGray = m_CurrImgGray;
        m_PrevRenderTime = m_CurrRanderTime;
        m_PrevKeypoints = Keypoints2Point2f(cv2.Cv2.FAST(m_CurrImgGray, m_FastThreshold));
        m_IsReady = true;
    }

    /// <summary>
    /// Compute the divergence (D) wich is the empirical estimation of D = v_y/h; where v_y is the velocity on the y axis and h is the altitude.
    /// </summary>
    /// <returns> The empirical estimation of the divergence </returns>
    private float UpdateDivergence() {
        float D_hat = 0f;
        int N_D = Mathf.Clamp(m_MaxDivergencePoints, 0, Mathf.Min(m_PrevKeypoints.Length, m_CurrKeypoints.Length));
        if (N_D == 0) {
            m_Divergence = 0f;
            return m_Divergence;
        }

        float[] prevDistances = new float[N_D];
        float[] currDistances = new float[N_D];

        for (int i = 0; i < N_D; i++) {
            prevDistances[i] = (float)m_PrevKeypoints[i].DistanceTo(m_PrevKeypoints[(m_PrevKeypoints.Length - 1) - i]);
            currDistances[i] = (float)m_CurrKeypoints[i].DistanceTo(m_CurrKeypoints[(m_CurrKeypoints.Length - 1) - i]);

            if (Mathf.Abs(prevDistances[i]) > 0.01f) {
                D_hat += (currDistances[i] - prevDistances[i]) / prevDistances[i];
            }
        }
        m_Divergence = D_hat / (N_D * GetIntegrationTime());
        if (float.IsNaN(m_Divergence)) {
            m_Divergence = 0f;
        }
        else if (float.IsInfinity(m_Divergence)) {
            m_Divergence = 100f;
        }
        return m_Divergence;
    }

    /// <summary>
    /// Compute the current keypoints from the current frame. The keypoints are found with the pyramidal 
    /// kenede tracker using the keypoints of the previous frame found with the Fast algorithm.
    /// </summary>
    /// <returns> true if the computation is done correctly else return false. </returns>
    private bool ComputeCurrentKeypoints() {
        bool valid = m_PrevKeypoints.Length > 0;
        if (!valid) {
            return valid;
        }
        m_CurrKeypoints = new cv2.Point2f[m_PrevKeypoints.Length];
        byte[] status;
        float[] err;
        cv2.Cv2.CalcOpticalFlowPyrLK(m_PrevImgGray, m_CurrImgGray, m_PrevKeypoints, ref m_CurrKeypoints, out status, out err);
        return valid;
    }


    public cv2.Point2f[] Keypoints2Point2f(cv2.KeyPoint[] keypoints) {
        cv2.Point2f[] points = new cv2.Point2f[keypoints.Length];
        for (int i = 0; i < keypoints.Length; i++) {
            points[i] = keypoints[i].Pt;
        }
        return points;
    }

    public float GetIntegrationTime() {
        return m_CurrRanderTime - m_PrevRenderTime;
    }


    public static void RawInputToTexture(Camera obsCamera, Texture2D texture2D) {
        if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null) {
            Debug.LogError("GraphicsDeviceType is Null. This will likely crash when trying to render.");
        }

        Rect rect = obsCamera.rect;
        obsCamera.rect = new Rect(0f, 0f, 1f, 1f);
        int depthBuffer = 24;
        RenderTextureFormat format = RenderTextureFormat.Default;
        RenderTextureReadWrite readWrite = RenderTextureReadWrite.Default;
        RenderTexture temporary = RenderTexture.GetTemporary(texture2D.width, texture2D.height, depthBuffer, format, readWrite);
        RenderTexture active = RenderTexture.active;
        RenderTexture targetTexture = obsCamera.targetTexture;
        RenderTexture.active = temporary;
        obsCamera.targetTexture = temporary;
        obsCamera.Render();
        texture2D.ReadPixels(new Rect(0f, 0f, texture2D.width, texture2D.height), 0, 0);
        obsCamera.targetTexture = targetTexture;
        obsCamera.rect = rect;
        RenderTexture.active = active;
        RenderTexture.ReleaseTemporary(temporary);
    }

    public int GetOutputSize() {
        return m_DivergenceAsOneHot ? m_DivergenceBins : 1;
    }

    public int GetDivergenceBinIndex(float divergence) {
        // [-infty, binSize, binSize*2, binSize*3, ..., binSize*bins, infty]
        float value = divergence / m_DivergenceBinSize;
        int n = (int)(m_DivergenceBins / 2);
        value = Mathf.Clamp(value, -n, n);
        int binIndex = (int)(value + n);
        return binIndex;
    }

    private float[] GetDivergenceAsOneHotVector() {
        float[] oneHot = new float[m_DivergenceBins];
        int bin = GetDivergenceBinIndex(m_Divergence);
        for (int i = 0; i < m_DivergenceBins; i++) {
            oneHot[i] = (i == bin) ? 1f : 0f;
        }
        return oneHot;
    }

    public float[] GetObservationList() {
        if (m_DivergenceAsOneHot) {
            return GetDivergenceAsOneHotVector();
        }
        return new float[] { m_Divergence };
    }

}






[RequireComponent(typeof(Camera))]
public class DivergenceCamera : SensorComponent, IDisposable {

    private EnvironmentParameters environmentParameters;
    private Camera m_CameraInput;
    private DivergenceSensor m_Sensor;
    [SerializeField] private string sensorName = "DivergenceSensor";

    [Header("Processing Parameters")]
    [SerializeField] private int maxDivergencePoints = 100;
    [Range(0, 255)] [SerializeField] private int fastThreshold = 10;
    [Range(1, 1000)][SerializeField] private int observationStacks = 10;
    [SerializeField] private bool divergenceAsOneHot = true;
    [SerializeField] private int divergenceBins = 20;
    [SerializeField] private float divergenceBinSize = 10f;

    [Header("Processing references")]
    private cv2.Mat prevImgGray;
    private cv2.Point2f[] prevKeypoints;
    private cv2.Mat currImgGray;
    private cv2.Point2f[] currKeypoints;
    private float prevRenderTime = 0f;
    private float currRanderTime = 0f;
    private float m_divergence = 0f;
    private bool isReady = false;


    // Start is called before the first frame update
    void Start()
    {
        SetParameters();
        SetCamera();
    }


    private void SetCamera() {
        m_CameraInput = GetComponent<Camera>();
    }

    private void SetParameters() {
        environmentParameters = Academy.Instance.EnvironmentParameters;
        observationStacks = (int)environmentParameters.GetWithDefault("observationStacks", (float)observationStacks);
        divergenceAsOneHot = AsBool(environmentParameters.GetWithDefault("divergenceAsOneHot", System.Convert.ToSingle(divergenceAsOneHot)));
        divergenceBins = (int)environmentParameters.GetWithDefault("divergenceBins", (float)divergenceBins);
        divergenceBinSize = environmentParameters.GetWithDefault("divergenceBinSize", divergenceBinSize);
        fastThreshold = (int)environmentParameters.GetWithDefault("fastThreshold", (float)fastThreshold);
        maxDivergencePoints = (int)environmentParameters.GetWithDefault("maxDivergencePoints", (float)maxDivergencePoints);
    }


    public override ISensor[] CreateSensors() {
        Dispose();
        SetParameters();
        SetCamera();
        m_Sensor = new DivergenceSensor(
            m_CameraInput, 
            sensorName, 
            p_FastThreshold: fastThreshold, 
            p_MaxDivergencePoints: maxDivergencePoints,
            p_DivergenceAsOneHot: divergenceAsOneHot,
            p_DivergenceBins: divergenceBins,
            p_DivergenceBinSize: divergenceBinSize,
            p_IsStacked: observationStacks != 1
        );

        if (observationStacks != 1) {
            return new ISensor[] { new StackingSensor(m_Sensor, observationStacks) };
        }
        return new ISensor[] { m_Sensor };
    }

    /// <summary>
    /// Clean up the sensor created by CreateSensors().
    /// </summary>
    public void Dispose() {
        if (!ReferenceEquals(m_Sensor, null)) {
            m_Sensor.Dispose();
            m_Sensor = null;
        }
    }

    void OnDisabled() {
        Dispose();
    }

    public static bool AsBool(float value) {
        return Mathf.Approximately(Mathf.Min(value, 1), 1);
    }

}
