using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using cv2 = OpenCvSharp;


[RequireComponent(typeof(Camera))]
public class DivergenceCamera : MonoBehaviour
{
    private Camera divergenceCamera;

    [Header("Processing Parameters")]
    [SerializeField] private int maxDivergencePoints = 1000;
    [Range(0, 255)] [SerializeField] private int fastThreshold = 10;

    [Header("Processing references")]
    private cv2.Mat prevImgGray;
    private cv2.Point2f[] prevKeypoints;
    private cv2.Mat currImgGray;
    private cv2.Point2f[] currKeypoints;
    private float prevRenderTime = 0f;
    private float currRanderTime = 0f;
    private float divergence = 0f;
    private bool isReady = false;


    // Start is called before the first frame update
    void Start()
    {
        divergenceCamera = GetComponent<Camera>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public bool IsReady() {
        return isReady;
    }

    public float GetDivergence() {
        return divergence;
    }

    public float UpdateState() {
        UpdateCurrentFrame();
        if (isReady) {
            bool valid = ComputeCurrentKeypoints();
            if (valid) UpdateDivergence();
        }
        UpdatePreviousKeypoints();
        return divergence;
    }


    /// <summary>
    /// Update the current states of the camera. 
    ///    -> Render the camera manualy.
    ///    -> update the current render time.
    ///    -> update currImgGray that contain the current frame in grayscale color.
    /// </summary>
    private void UpdateCurrentFrame() {
        RenderTexture activeRenderTexture = RenderTexture.active;
        RenderTexture.active = divergenceCamera.targetTexture;

        divergenceCamera.Render();
        currRanderTime = Time.time * Time.timeScale;

        Texture2D image = new Texture2D(divergenceCamera.pixelWidth, divergenceCamera.pixelHeight);
        image.ReadPixels(new Rect(0, 0, divergenceCamera.pixelWidth, divergenceCamera.pixelHeight), 0, 0);
        image.Apply();
        RenderTexture.active = activeRenderTexture;

        cv2.Mat img = cv2.Unity.TextureToMat(image);
        Destroy(image);
        currImgGray = new cv2.Mat();
        cv2.Cv2.CvtColor(img, currImgGray, cv2.ColorConversionCodes.BGR2GRAY);
    }

    /// <summary>
    /// The current states go to the previous states.
    ///     -> previous frame become the current frame
    ///     -> previous render time become the current render time.
    ///     -> the previous keypoints are compute with the FASt algorithm from the current frame
    ///     -> set the isReady flag to true
    /// </summary>
    private void UpdatePreviousKeypoints() {
        prevImgGray = currImgGray;
        prevRenderTime = currRanderTime;
        prevKeypoints = Keypoints2Point2f(cv2.Cv2.FAST(currImgGray, fastThreshold));
        isReady = true;
    }

    /// <summary>
    /// Compute the divergence (D) wich is the empirical estimation of D = v_y/h; where v_y is the velocity on the y axis and h is the altitude.
    /// </summary>
    /// <returns> The empirical estimation of the divergence </returns>
    private float UpdateDivergence() {
        float D_hat = 0f;
        int N_D = Mathf.Clamp(maxDivergencePoints, 0, Mathf.Min(prevKeypoints.Length, currKeypoints.Length));
        if (N_D == 0) {
            divergence = 0f;
            return divergence;
        }

        float[] prevDistances = new float[N_D];
        float[] currDistances = new float[N_D];

        for (int i = 0; i < N_D; i++) {
            prevDistances[i] = (float)prevKeypoints[i].DistanceTo(prevKeypoints[(prevKeypoints.Length - 1) - i]);
            currDistances[i] = (float)currKeypoints[i].DistanceTo(currKeypoints[(currKeypoints.Length - 1) - i]);

            if (Mathf.Abs(prevDistances[i]) > 0.01f) {
                D_hat += (currDistances[i] - prevDistances[i]) / prevDistances[i];
            }
        }
        divergence = D_hat / (N_D * GetIntegrationTime());
        if (float.IsNaN(divergence)) {
            divergence = 0f;
        }
        else if (float.IsInfinity(divergence)) {
            divergence = 100f;
        }
        return divergence;
    }

    /// <summary>
    /// Compute the current keypoints from the current frame. The keypoints are found with the pyramidal 
    /// kenede tracker using the keypoints of the previous frame found with the Fast algorithm.
    /// </summary>
    /// <returns> true if the computation is done correctly else return false. </returns>
    private bool ComputeCurrentKeypoints() {
        bool valid = prevKeypoints.Length > 0;
        if (!valid) {
            return valid;
        }
        currKeypoints = new cv2.Point2f[prevKeypoints.Length];
        byte[] status;
        float[] err;
        cv2.Cv2.CalcOpticalFlowPyrLK(prevImgGray, currImgGray, prevKeypoints, ref currKeypoints, out status, out err);
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
        return currRanderTime - prevRenderTime;
    }

}
