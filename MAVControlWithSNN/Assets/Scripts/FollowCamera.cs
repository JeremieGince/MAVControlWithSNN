using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.MLAgents;

public class FollowCamera : MonoBehaviour
{
    private EnvironmentParameters environmentParameters;
    [SerializeField] private EnvironmentManager environmentManager;
    public Transform target;
    public Vector3 offset = new Vector3(0, 2, -3);
    public Vector3 rotation = new Vector3(25, 0, 0);

    private Camera m_camera;
    public GameObject canvas;
    public RawImage neuromorphicRawImage;
    public RawImage frontCameraRawImage;
    private Texture2D texture;
    private NeuromorphicCamera neuroCam;
    [SerializeField] private bool followTargetAgent = true;
    private bool isInPlace = false;


    // Start is called before the first frame update
    void Start()
    {
        m_camera = GetComponent<Camera>();
        neuroCam = target?.GetComponentInChildren<NeuromorphicCamera>();
        environmentParameters = Academy.Instance.EnvironmentParameters;
        followTargetAgent = AsBool(environmentParameters.GetWithDefault("camFollowTargetAgent", System.Convert.ToSingle(followTargetAgent)));
    }

    // Update is called once per frame
    void Update()
    {
        if (followTargetAgent) {
            FollowTargetAgent();
            isInPlace = false;
        }
        else {
            canvas.SetActive(false);
            if(!isInPlace) LookAtEnvironments();
        }
        
        
    }

    public void FollowTargetAgent() {
        canvas.SetActive(true);
        if (target != null) {
            transform.position = target.position + offset;
            transform.rotation = Quaternion.Euler(rotation.x, rotation.y, rotation.z);
        }

        if (neuroCam != null) {
            neuromorphicRawImage.texture = neuroCam.GetStateAsTexture();
            frontCameraRawImage.texture = neuroCam.GetRawInput();

        }
    }


    public void LookAtEnvironments() {
        Vector3 meanPos = Vector3.zero;
        List<EnvironmentScript> envScripts = environmentManager.GetEnvironments();
        for (int i = 0; i < envScripts.Count; i++) {
            meanPos = (i * meanPos + envScripts[i].transform.position) / (i + 1);
        }
        Vector3 meanGroundPos = new Vector3(meanPos.x, 0f, meanPos.z);
        transform.position = meanPos;
        transform.LookAt(meanGroundPos);
        for (int i = 0; i < 1000; i++) {
            
            if (AllEnvironmentsInView()) {
                isInPlace = true;
                transform.position += -transform.forward;
                transform.LookAt(meanGroundPos);
                break;
            }

            transform.position += -transform.forward;
            transform.LookAt(meanGroundPos);

        }
    }



    public bool AllEnvironmentsInView() {
        List<EnvironmentScript> envScripts = environmentManager.GetEnvironments();
        for (int i = 0; i < envScripts.Count; i++) {
            foreach(Vector3 pos in envScripts[i].GetFloorExtBounds()) {
                Vector3 viewPos = m_camera.WorldToViewportPoint(pos);
                bool inView = viewPos.x >= 0 && viewPos.x <= 1 && viewPos.y >= 0 && viewPos.y <= 1 && viewPos.z > 0;
                if (!inView) {
                    return false;
                }
            }
        }
        return true;
    }



    public static bool AsBool(float value) {
        return Mathf.Approximately(Mathf.Min(value, 1), 1);
    }





}
