using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Properler : MonoBehaviour
{

    [SerializeField] private Transform rotationTransform;
    [SerializeField] private Rigidbody droneRigidbody;
    public float maxForce = 10f;
    public float rotationSpeed = 300f;
    public float constForce = 0f;
    [SerializeField] private float currForce = 0f;
    [Range(0f, 1f)] [SerializeField] private float currForceRatio = 0f;
    public bool enableTorque = true;

    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        rotationTransform.RotateAround(transform.position, transform.up, Time.deltaTime * rotationSpeed);
    }

    private void FixedUpdate() {
        AddForce(constForce);

        if (enableTorque) {
            //droneRigidbody.AddForceAtPosition(Time.fixedDeltaTime * currForce * transform.up, transform.position);
        }
        else {
            //droneRigidbody.AddForce(Time.fixedDeltaTime * currForce * transform.up, ForceMode.Impulse);
        }
        AddForce(currForce);
    }



    public void SetForce(float newForce) {
        currForce = newForce;
        currForceRatio = newForce / maxForce;
    }

    public void SetForceRatio(float newForceRatio) {
        newForceRatio = Mathf.Clamp(newForceRatio, -1f, 1f);
        currForce = newForceRatio * maxForce;
        currForceRatio = newForceRatio;
    }


    public void AddForce(float force) {
        if (enableTorque) {
            droneRigidbody.AddForceAtPosition(force * transform.up, transform.position, ForceMode.Force);
        }
        else {
            droneRigidbody.AddForce(force * transform.up, ForceMode.Force);
        }
        
    }


    public void AddForceRatio(float forceRatio) {
        forceRatio = Mathf.Clamp(forceRatio, -1f, 1f);
        if (enableTorque) {
            droneRigidbody.AddForceAtPosition(Time.deltaTime * maxForce * forceRatio * transform.up, transform.position);
        }
        else {
            droneRigidbody.AddForce(Time.deltaTime * maxForce * forceRatio * transform.up);
        }
    }




}
