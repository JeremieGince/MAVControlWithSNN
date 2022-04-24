using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class FollowCamera : MonoBehaviour
{

    public Transform target;
    public Vector3 offset = new Vector3(0, 2, -3);
    public Vector3 rotation = new Vector3(25, 0, 0);

    public RawImage neuromorphicRawImage;
    public RawImage frontCameraRawImage;
    private Texture2D texture;
    private NeuromorphicCamera neuroCam;


    // Start is called before the first frame update
    void Start()
    {
        neuroCam = target?.GetComponentInChildren<NeuromorphicCamera>();
    }

    // Update is called once per frame
    void Update()
    {
        if(target != null) {
            transform.position = target.position + offset;
            transform.rotation = Quaternion.Euler(rotation.x, rotation.y, rotation.z);
        }

        if (neuroCam != null) {
            neuromorphicRawImage.texture = neuroCam.GetStateAsTexture();
            frontCameraRawImage.texture = neuroCam.GetRawInput();
            
        }
        
    }
}
