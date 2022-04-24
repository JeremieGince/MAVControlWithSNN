using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(RigidTest))]
public class LevelScriptEditor : Editor {
    public override void OnInspectorGUI() {
        RigidTest myTarget = (RigidTest)target;

        DrawDefaultInspector();

        if (GUILayout.Button("Apply")){
            myTarget.ApplyForce();
        }
        if (GUILayout.Button("ResetRigidbody")) {
            myTarget.ResetRigidbody();
        }
        if (GUILayout.Button("ResetPosition")) {
            myTarget.ResetPosition();
        }
        if (GUILayout.Button("ResetForce")) {
            myTarget.ResetForce();
        }
        if (GUILayout.Button("ResetAll")) {
            myTarget.ResetAll();
        }
    }
}


public class RigidTest : MonoBehaviour
{

    [SerializeField] private Rigidbody rigidbody;
    public Vector3 fixedDir = Vector3.up;
    public ForceMode fixedMode = ForceMode.Force;
    public float fixedForce = 0f;
    public bool multiplyByDt = true;
    public Vector3 forcedDir = Vector3.up;
    public ForceMode forceMode = ForceMode.Force;
    public float force = 0f;

    private Vector3 startPosition;

    // Start is called before the first frame update
    void Start()
    {
        rigidbody = GetComponent<Rigidbody>();
        startPosition = transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        
    }



    private void FixedUpdate() {
        if (multiplyByDt) {
            rigidbody.AddForce(fixedForce * fixedDir * Time.fixedDeltaTime, fixedMode);
        }
        else {
            rigidbody.AddForce(fixedForce * fixedDir, fixedMode);
        }
        
    }


    public void ApplyForce() {
        rigidbody.AddForce(force * forcedDir, forceMode);
    }

    public void ResetRigidbody() {
        rigidbody.velocity = Vector3.zero;
        rigidbody.angularVelocity = Vector3.zero;
    }

    public void ResetForce() {
        fixedForce = 0f;
        force = 0f;
    }

    public void ResetPosition() {
        transform.position = startPosition;
        transform.eulerAngles = Vector3.zero;
    }

    public void ResetAll() {
        ResetRigidbody();
        ResetForce();
        ResetPosition();
    }




}
