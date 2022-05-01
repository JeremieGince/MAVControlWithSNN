using System;
using System.Threading.Tasks;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Rendering;
using cv2 = OpenCvSharp;

public class NeuromorphicCameraSensor : ISensor, IDisposable {

    private Camera m_CameraInput;
    private Texture2D m_TextureInput;
    private string m_SensorName;

    private int m_Width = 84;
    private int m_Height = 84;
    private int m_Threshold;

    private cv2.Mat m_PrevImgGray;
    private cv2.Mat m_CurrImgGray;
    private cv2.Mat m_PrevResizeImg;
    private cv2.Mat m_CurrResizeImg;
    private float m_PrevRenderTime = 0f;
    private float m_CurrRanderTime = 0f;
    private cv2.Mat m_State;
    private Texture2D m_StateTexture;

    private bool m_IsReady = false;

    public NeuromorphicCameraSensor(Camera p_cameraInput, string p_sensorName, int p_width, int p_height, int p_threshold) {
        m_CameraInput = p_cameraInput;
        m_SensorName = p_sensorName;
        m_Width = p_width;
        m_Height = p_height;
        m_Threshold = p_threshold;

        m_TextureInput = new Texture2D(m_CameraInput.pixelWidth, m_CameraInput.pixelHeight, TextureFormat.RGB24, mipChain: false);
        m_State = cv2.Mat.Zeros(rows: m_Height, cols: m_Width, type: cv2.MatType.CV_32F);
        m_StateTexture = new Texture2D(m_Width, m_Height, TextureFormat.RGBA32, mipChain: false);
        m_IsReady = false;
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
        m_CurrResizeImg = new cv2.Mat(rows: m_Height, cols: m_Width, type: cv2.MatType.CV_32F);
        cv2.Cv2.CvtColor(img, m_CurrImgGray, cv2.ColorConversionCodes.BGR2GRAY);
        cv2.Cv2.Resize(m_CurrImgGray, m_CurrResizeImg, new cv2.Size(m_Width, m_Height));
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


    private void UpdatePreviousFrame() {
        m_PrevResizeImg = m_CurrResizeImg;
        m_PrevImgGray = m_CurrImgGray;
        m_PrevRenderTime = m_CurrRanderTime;
        m_IsReady = true;
    }

    private void ComputeCurrentState() {
        cv2.Mat diff = new cv2.Mat(rows: m_Height, cols: m_Width, type: cv2.MatType.CV_32F);
        cv2.Cv2.Subtract(m_CurrResizeImg, m_PrevResizeImg, diff);

        cv2.Mat diff_64F = new cv2.Mat(rows: m_Height, cols: m_Width, type: cv2.MatType.CV_64F);
        diff.ConvertTo(diff_64F, cv2.MatType.CV_64F);

        cv2.Mat log_mat = new cv2.Mat(rows: m_Height, cols: m_Width, type: cv2.MatType.CV_64F);
        cv2.Cv2.Log(cv2.Cv2.Abs(diff_64F), log_mat);

        cv2.Cv2.Threshold(log_mat, m_State, m_Threshold, 255f, cv2.ThresholdTypes.Binary);
    }

    public float GetIntegrationTime() {
        return m_CurrRanderTime - m_PrevRenderTime;
    }

    byte[] ISensor.GetCompressedObservation() {
        StateToTexture(m_State, m_StateTexture);
        return m_StateTexture.EncodeToPNG();
    }

    CompressionSpec ISensor.GetCompressionSpec() {
        return new CompressionSpec(SensorCompressionType.None);
    }

    string ISensor.GetName() {
        return m_SensorName;
    }

    ObservationSpec ISensor.GetObservationSpec() {
        return ObservationSpec.Visual(m_Height, m_Width, 1);
    }

    void ISensor.Reset() {
        //Dispose();
        m_State = cv2.Mat.Zeros(rows: m_Height, cols: m_Width, type: cv2.MatType.CV_32F);
    }

    void ISensor.Update() {
        UpdateCurrentFrame();
        if(m_IsReady) ComputeCurrentState();
        UpdatePreviousFrame();
    }

    int ISensor.Write(ObservationWriter writer) {
        StateToTexture(m_State, m_StateTexture);
        writer.WriteTexture(m_StateTexture, true);
        return m_State.Height * m_State.Width;
    }

    public void Dispose() {
        m_PrevImgGray.Dispose();
        m_CurrImgGray.Dispose();
        m_PrevResizeImg.Dispose();
        m_CurrResizeImg.Dispose();
        m_State.Dispose();

        if ((object)m_TextureInput != null) {
            DestroyTexture(m_TextureInput);
            m_TextureInput = null;
        }
        if ((object)m_StateTexture != null) {
            DestroyTexture(m_StateTexture);
            m_StateTexture = null;
        }
    }
    
    public static void DestroyTexture(Texture2D texture) {
        if (Application.isEditor) {
            UnityEngine.Object.DestroyImmediate(texture);
        }
        else {
            UnityEngine.Object.Destroy(texture);
        }
    }


    public Texture2D GetStateAsTexture() {
        //cv2.Mat stateResized = new cv2.Mat(rows: 256, cols: 256, type: cv2.MatType.CV_32F);
        //cv2.Cv2.Resize(m_State, stateResized, new cv2.Size(m_Width, m_Height));
        StateToTexture(m_State, m_StateTexture);
        return m_StateTexture;
    }


    public Texture2D GetRawInputTexture() {
        RawInputToTexture(m_CameraInput, m_TextureInput);
        return m_TextureInput;
    }


    public static Texture2D StateToTexture(cv2.Mat sourceMat, Texture2D texture2D) {
        //Get the height and width of the Mat 
        int imgHeight = sourceMat.Height;
        int imgWidth = sourceMat.Width;

        byte[] matData = new byte[imgHeight * imgWidth];
        cv2.Mat sourceMat_8U = new cv2.Mat(rows: imgHeight, cols: imgWidth, type: cv2.MatType.CV_8U);
        sourceMat.ConvertTo(sourceMat_8U, cv2.MatType.CV_8U);
        //Get the byte array and store in matData
        sourceMat_8U.GetArray(0, 0, matData);
        
        //Create the Color array that will hold the pixels 
        Color32[] c = new Color32[imgHeight * imgWidth];

        //Get the pixel data from parallel loop
        Parallel.For(0, imgHeight, i => {
            for (var j = 0; j < imgWidth; j++) {
                byte vec = matData[j + ((imgHeight-1) - i) * imgWidth];
                Color32 color32 = new Color32 {
                    r = vec,
                    g = vec,
                    b = vec,
                    a = 255
                };
                c[j + i * imgWidth] = color32;
            }
        });

        //Create Texture from the result
        //Texture2D tex = new Texture2D(imgWidth, imgHeight, TextureFormat.RGBA32, true, true);
        //tex.SetPixels32(c);
        //tex.Apply();
        //return tex;
        texture2D.SetPixels32(c);
        texture2D.Apply();
        return texture2D;
    }

}





[RequireComponent(typeof(Camera))]
public class NeuromorphicCamera : SensorComponent, IDisposable {

    private EnvironmentParameters environmentParameters;

    private Camera cameraInput;
    private Texture2D m_TextureInput;
    private NeuromorphicCameraSensor sensor;

    [Header("Processing")]
    [SerializeField] private string sensorName = "NeuromorphicCamera";
    [SerializeField] private int width = 84;
    [SerializeField] private int height = 84;
    [SerializeField] private int threshold = 1;
    [Range(1, 1000)][SerializeField] private int observationStacks = 10;


    private void Reset() {
        SetCamera();
    }

    // Start is called before the first frame update
    void Start()
    {
        SetParameters();
        SetCamera();
        m_TextureInput = new Texture2D(cameraInput.pixelWidth, cameraInput.pixelHeight, TextureFormat.RGBA32, mipChain: false);
    }


    private void SetCamera() {
        cameraInput = GetComponent<Camera>();
        //cameraInput.targetTexture = new RenderTexture(cameraInput.targetTexture);
    }

    private void SetParameters() {
        environmentParameters = Academy.Instance.EnvironmentParameters;
        observationStacks = (int)environmentParameters.GetWithDefault("observationStacks", (float)observationStacks);
        width = (int)environmentParameters.GetWithDefault("observationWidth", (float)width);
        height = (int)environmentParameters.GetWithDefault("observationHeight", (float)height);
    }

    // Update is called once per frame
    void Update()
    {
    }

    /*
    public cv2.Mat GetStateAsMat() {
        return state;
    }
    */

    public Texture2D GetStateAsTexture() {
        return sensor?.GetStateAsTexture();
    }

    public Texture2D GetRawInput() {
        Texture2D rawInput = sensor?.GetRawInputTexture();
        m_TextureInput.SetPixels(rawInput.GetPixels());
        m_TextureInput.Apply();
        return m_TextureInput;
    }


    public override ISensor[] CreateSensors() {
        Dispose();
        SetParameters();
        SetCamera();
        sensor = new NeuromorphicCameraSensor(cameraInput, sensorName, width, height, threshold);

        if (observationStacks != 1) {
            return new ISensor[] { new StackingSensor(sensor, observationStacks) };
        }
        return new ISensor[] { sensor };
    }

    /// <summary>
    /// Clean up the sensor created by CreateSensors().
    /// </summary>
    public void Dispose() {
        if (!ReferenceEquals(sensor, null)) {
            sensor.Dispose();
            sensor = null;
        }
    }

    void OnDisabled() {
        Dispose();
    }

}
