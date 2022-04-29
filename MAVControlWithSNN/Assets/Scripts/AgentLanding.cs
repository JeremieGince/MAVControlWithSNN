using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;
using cv2 = OpenCvSharp;



public enum WinState { Win, HalfWin, Lose }


public class AgentLanding : Agent
{
    private EnvironmentParameters environmentParameters;

    [Header("Rewards")]
    public float stepRewardFactor = 0f;
    public float reachGoalReward = 1f;
    public float hitWallReward = -1f;

    [Header("Environment Parameters")]
    [SerializeField] private string behaviorName = "Landing";
    [SerializeField] private bool enableTorque = true;
    [SerializeField] private bool enableDisplacement = true;
    [SerializeField] private bool enableDivergence = true;
    [SerializeField] private bool enableNeuromorphicCamera = true;
    [SerializeField] private bool enableCamera = true;
    [SerializeField] private float divergenceSetPoint = 1f;
    [SerializeField] private float droneMinStartY = 1f;
    [SerializeField] private float droneMaxStartY = 100f;
    [SerializeField] private bool usePositionAsInput = true;
    [SerializeField] private bool useRotationAsInput = true;
    [SerializeField] private bool useVelocityAsInput = true;
    [SerializeField] private bool useAngularVelocityAsInput = true;
    [SerializeField] private float propelersConstForce = 0f;
    [SerializeField] private bool useForceRatio = true;
    [SerializeField] private float properlersForce;
    [SerializeField] private float droneDrag = 1f;
    [SerializeField] private float droneAngularDrag = 5f;
    [SerializeField] private float targetLandingVelocity = 1f;
    [SerializeField] private float targetLandingAngularVelocity = 5f;
    [SerializeField] private float targetLandingAngleMagnitude = 15f;

    [Header("Processing Parameters")]
    [SerializeField] private bool divergenceAsOneHot = true;
    [SerializeField] private int divergenceBins = 20;
    [SerializeField] private int divergenceBinSize = 10;


    [Header("Propelers")]
    [SerializeField] private Properler rotorFL;
    [SerializeField] private Properler rotorFR;
    [SerializeField] private Properler rotorBL;
    [SerializeField] private Properler rotorBR;

    [Header("Materials")]
    [SerializeField] private Material reachGoalMaterial;
    [SerializeField] private Material halfWinMaterial;
    [SerializeField] private Material hitWallMaterial;
    [SerializeField] private Material defaultMaterial;
    [SerializeField] private MeshRenderer floorMeshRenderer;

    [Header("Drone references")]
    [SerializeField] private CameraSensorComponent cameraSensor;
    [SerializeField] private DivergenceCamera divergenceCamera;
    [SerializeField] private NeuromorphicCamera neuromorphicCamera;
    private Rigidbody droneRigidbody;
    private Vector3 droneSize;


    [Header("Environment References")]
    [SerializeField] private EnvironmentScript environmentScript;
    [SerializeField] private Transform ceilling;
    [SerializeField] private Transform targetTransform;
    private BehaviorParameters behaviorParameters;
    private Vector3 startPosition;
    

    private void Awake() {
        droneRigidbody = GetComponent<Rigidbody>();
        properlersForce = 2f * 9.81f * droneRigidbody.mass;
        droneSize = droneRigidbody.GetComponentInChildren<Renderer>().bounds.size;

        InitializeEnvironmentParameters();
        InitializeDroneState();
        UpdateBehaviorParameters();
    }

    private void Start() {
        droneRigidbody = GetComponent<Rigidbody>();
        droneSize = droneRigidbody.GetComponentInChildren<Renderer>().bounds.size;
        droneRigidbody.velocity = Vector3.zero;
        droneRigidbody.angularVelocity = Vector3.zero;
        //droneRigidbody.AddForce(droneRigidbody.mass * 9.8f * droneRigidbody.transform.up);
        environmentScript = GetComponentInParent<EnvironmentScript>();

        InitializeEnvironmentParameters();
        InitializeDroneState();
        UpdateBehaviorParameters();
    }


    private void UpdateBehaviorParameters() {
        behaviorParameters = GetComponent<BehaviorParameters>();
        behaviorParameters.BehaviorName = behaviorName;
        if (enableTorque) {
            behaviorParameters.BrainParameters.ActionSpec = new ActionSpec(numContinuousActions: 4);
        }
        else {
            behaviorParameters.BrainParameters.ActionSpec = new ActionSpec(numContinuousActions: 1);
        }

        int spaceSize = 0;
        spaceSize += enableDisplacement ? 3 : 0;
        spaceSize += usePositionAsInput ? 3 : 0;
        spaceSize += useRotationAsInput ? 3 : 0;
        spaceSize += useVelocityAsInput ? 3 : 0;
        spaceSize += useAngularVelocityAsInput ? 3 : 0;
        if (enableDivergence) {
            spaceSize += divergenceAsOneHot ? divergenceBins : 1;
        }
        behaviorParameters.BrainParameters.VectorObservationSize = spaceSize;
        behaviorParameters.BrainParameters.NumStackedVectorObservations = 1;
    }


    public void SetEnableTorque(bool newValue) {
        enableTorque = newValue;
        rotorFL.enableTorque = newValue;
        rotorFR.enableTorque = newValue;
        rotorBL.enableTorque = newValue;
        rotorBR.enableTorque = newValue;
    }

    public void InitializePropelers() {
        SetEnableTorque(enableTorque);
        rotorFL.maxForce = properlersForce;
        rotorFR.maxForce = properlersForce;
        rotorBL.maxForce = properlersForce;
        rotorBR.maxForce = properlersForce;

        rotorFL.constForce = propelersConstForce / 4f;
        rotorFR.constForce = propelersConstForce / 4f;
        rotorBL.constForce = propelersConstForce / 4f;
        rotorBR.constForce = propelersConstForce / 4f;
    }


    private void InitializeEnvironmentParameters() {
        environmentParameters = Academy.Instance.EnvironmentParameters;

        // Rewards parameters
        stepRewardFactor = environmentParameters.GetWithDefault("stepRewardFactor", stepRewardFactor);
        reachGoalReward = environmentParameters.GetWithDefault("reachGoalReward", reachGoalReward);
        hitWallReward = environmentParameters.GetWithDefault("hitWallReward", hitWallReward);


        // Environments parameters
        enableTorque = AsBool(environmentParameters.GetWithDefault("enableTorque", System.Convert.ToSingle(enableTorque)));
        enableDisplacement = AsBool(environmentParameters.GetWithDefault("enableDisplacement", System.Convert.ToSingle(enableDisplacement)));
        enableDivergence = AsBool(environmentParameters.GetWithDefault("enableDivergence", System.Convert.ToSingle(enableDivergence)));
        enableNeuromorphicCamera = AsBool(environmentParameters.GetWithDefault("enableNeuromorphicCamera", System.Convert.ToSingle(enableNeuromorphicCamera)));
        enableCamera = AsBool(environmentParameters.GetWithDefault("enableCamera", System.Convert.ToSingle(enableCamera)));
        divergenceSetPoint = environmentParameters.GetWithDefault("divergenceSetPoint", divergenceSetPoint);
        droneMinStartY = environmentParameters.GetWithDefault("droneMinStartY", droneMinStartY);
        droneMaxStartY = environmentParameters.GetWithDefault("droneMaxStartY", droneMaxStartY);
        usePositionAsInput = AsBool(environmentParameters.GetWithDefault("usePositionAsInput", System.Convert.ToSingle(usePositionAsInput)));
        useRotationAsInput = AsBool(environmentParameters.GetWithDefault("useRotationAsInput", System.Convert.ToSingle(useRotationAsInput)));
        useVelocityAsInput = AsBool(environmentParameters.GetWithDefault("useVelocityAsInput", System.Convert.ToSingle(useVelocityAsInput)));
        useAngularVelocityAsInput = AsBool(environmentParameters.GetWithDefault("useAngularVelocityAsInput", System.Convert.ToSingle(useAngularVelocityAsInput)));
        propelersConstForce = environmentParameters.GetWithDefault("propelersConstForce", propelersConstForce);
        useForceRatio = AsBool(environmentParameters.GetWithDefault("useForceRatio", System.Convert.ToSingle(useForceRatio)));
        properlersForce = environmentParameters.GetWithDefault("properlersForce", properlersForce);
        droneDrag = environmentParameters.GetWithDefault("droneDrag", droneDrag);
        droneAngularDrag = environmentParameters.GetWithDefault("droneAngularDrag", droneAngularDrag);
        targetLandingVelocity = environmentParameters.GetWithDefault("targetLandingVelocity", targetLandingVelocity);
        targetLandingAngularVelocity = environmentParameters.GetWithDefault("targetLandingAngularVelocity", targetLandingAngularVelocity);
        targetLandingAngleMagnitude = environmentParameters.GetWithDefault("targetLandingAngleMagnitude", targetLandingAngleMagnitude);


        // Processing parameters
        divergenceAsOneHot = AsBool(environmentParameters.GetWithDefault("divergenceAsOneHot", System.Convert.ToSingle(divergenceAsOneHot)));
        divergenceBins = (int)environmentParameters.GetWithDefault("divergenceBins", (float)divergenceBins);
        divergenceBinSize = (int)environmentParameters.GetWithDefault("divergenceBinSize", (float)divergenceBinSize);
    }


    private void UpdateCameraSensorParameters() {
        cameraSensor.gameObject.SetActive(enableCamera);
        cameraSensor.ObservationStacks = (int)environmentParameters.GetWithDefault("observationStacks", (float)cameraSensor.ObservationStacks);
        cameraSensor.Width = (int)environmentParameters.GetWithDefault("observationWidth", (float)cameraSensor.Width);
        cameraSensor.Height = (int)environmentParameters.GetWithDefault("observationHeight", (float)cameraSensor.Height);
    }

    private void InitializeDroneState() {
        droneRigidbody.velocity = Vector3.zero;
        droneRigidbody.angularVelocity = Vector3.zero;
        transform.localPosition = new Vector3(0, Random.Range(droneMinStartY, droneMaxStartY), 0);
        transform.localRotation = Quaternion.Euler(0, 0, 0);

        droneRigidbody.angularDrag = droneAngularDrag;
        droneRigidbody.drag = droneDrag;

        InitializePropelers();
        startPosition = transform.localPosition;

        divergenceCamera.gameObject.SetActive(enableDivergence);
        neuromorphicCamera.gameObject.SetActive(enableNeuromorphicCamera);
        UpdateCameraSensorParameters();
    }

    public override void OnEpisodeBegin() {
        InitializeEnvironmentParameters();
        InitializeDroneState();
        UpdateBehaviorParameters();
        ceilling.localPosition = new Vector3(0f, startPosition.y + 2f*(droneSize.y + 1f), 0f);
        environmentScript?.Randomize();
        //Debug.Log("Episode Begin");

        //Debug.Log("NumContinuousActions: " + behaviorParameters.BrainParameters.ActionSpec.NumContinuousActions);
    }


    public override void CollectObservations(VectorSensor sensor) {
        if (usePositionAsInput) {
            sensor.AddObservation(droneRigidbody.transform.localPosition);
        }
        if (enableDisplacement) {
            sensor.AddObservation(targetTransform.localPosition - transform.localPosition);
        }
        if (useRotationAsInput) {
            sensor.AddObservation(transform.localRotation.eulerAngles / 360f);
        }
        if (useVelocityAsInput) {
            sensor.AddObservation(droneRigidbody.velocity);
        }
        if (useAngularVelocityAsInput) {
            sensor.AddObservation(droneRigidbody.angularVelocity);
        }

        if (enableDivergence) {
            if (divergenceAsOneHot) {
                float div = divergenceCamera.GetDivergence();
                sensor.AddOneHotObservation(GetDivergenceBinIndex(div), divergenceBins);
            }
            else {
                sensor.AddObservation(divergenceCamera.GetDivergence());
            }
        }

    }

    public override void OnActionReceived(ActionBuffers actions) {
        float[] forces;
        if (enableTorque) {
            forces = new float[] { actions.ContinuousActions[0], actions.ContinuousActions[1], actions.ContinuousActions[2], actions.ContinuousActions[3] };
        }
        else {
            forces = new float[] { actions.ContinuousActions[0] / 4f, actions.ContinuousActions[0] / 4f, actions.ContinuousActions[0] / 4f, actions.ContinuousActions[0] / 4f };
        }
        if (useForceRatio) {
            rotorFL.SetForceRatio(forces[0]);
            rotorFR.SetForceRatio(forces[1]);
            rotorBR.SetForceRatio(forces[2]);
            rotorBL.SetForceRatio(forces[3]);
        }
        else {
            rotorFL.SetForce(forces[0]);
            rotorFR.SetForce(forces[1]);
            rotorBR.SetForce(forces[2]);
            rotorBL.SetForce(forces[3]);
        }
    }

    private void Update() {
        //AddReward(stepRewardFactor * Mathf.Abs(divergence - divergenceSetPoint) * Time.deltaTime / divergenceSetPoint);
        //float err = divergenceSetPoint - Mathf.Abs(GetTrueDivergence());
        //float norm = divergenceSetPoint + Mathf.Abs(GetTrueDivergence());
        float err = -Mathf.Abs(droneRigidbody.velocity.y);
        float norm = 1f;
        AddReward(stepRewardFactor * Time.deltaTime * err / norm);
    }


    public override void Heuristic(in ActionBuffers actionsOut) {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        float F_g = droneRigidbody.mass * 9.81f;
        float h = droneRigidbody.transform.localPosition.y - targetTransform.localPosition.y;
        float h_0 = startPosition.y - targetTransform.localPosition.y;
        float outputValue = (h - 0.1f) * (0.98f * F_g - F_g) / h_0 + 0.995f * F_g;
        if (useForceRatio) outputValue /= properlersForce;
        //Debug.Log("outputValue: " + outputValue);
        switch (behaviorParameters.BrainParameters.ActionSpec.NumContinuousActions) {
            case 1:
                continuousActions[0] = outputValue;
                break;
            case 4:
                continuousActions[0] = outputValue / 4f; // Input.GetKey(KeyCode.Q) ? 10f : 0f;
                continuousActions[1] = outputValue / 4f; // Input.GetKey(KeyCode.W) ? 10f : 0f;
                continuousActions[2] = outputValue / 4f; // Input.GetKey(KeyCode.N) ? 10f : 0f;
                continuousActions[3] = outputValue / 4f; // Input.GetKey(KeyCode.M) ? 10f : 0f;
                break;
            default:
                break;
        }

    }



    public float GetAuxiliaryReward() {
        float velocityReward = -droneRigidbody.velocity.magnitude;
        float angularVelocityReward = -droneRigidbody.angularVelocity.magnitude;
        float rotationReward = -transform.rotation.eulerAngles.magnitude;
        return velocityReward + angularVelocityReward + rotationReward;
    }



    private void OnTriggerEnter(Collider other) {
        OnContactEnter(other);
    }


    private void OnCollisionEnter(Collision collision) {
        OnContactEnter(collision.collider, collision);

    }


    private WinState CheckReachGoal(Collider other) {
        WinState winState = WinState.Win;
        if (enableDisplacement) {
            if (other.TryGetComponent<Goal>(out Goal goal)) {
                winState = WinState.Win;
            }
            else if (other.TryGetComponent<Ground>(out Ground ground)) {
                winState = WinState.HalfWin;
            }
        }
        if (other.TryGetComponent<Wall>(out Wall wall)) {
            winState = WinState.Lose;
        }
        return winState;
    }


    private void OnContactEnter(Collider other, Collision collision = null) {
        WinState winState = CheckReachGoal(other);
        float relativeVelocity = collision == null ? Mathf.Abs(droneRigidbody.velocity.magnitude) : Mathf.Abs(collision.relativeVelocity.magnitude);
        float angularVelocity = Mathf.Abs(droneRigidbody.angularVelocity.magnitude);
        float angleMagnitude = Mathf.Abs(droneRigidbody.transform.rotation.eulerAngles.magnitude);

        if (relativeVelocity > targetLandingVelocity || angularVelocity > targetLandingAngularVelocity || angleMagnitude > targetLandingAngleMagnitude) {
            if (enableDisplacement) {
                switch (winState) {
                    case WinState.Win:
                        winState = WinState.HalfWin;
                        break;
                    default:
                        winState = WinState.Lose;
                        break;
                }
            }
            else {
                winState = WinState.Lose;
            }            
        }
        switch (winState) {
            case WinState.Win:
                SetReward(reachGoalReward);
                floorMeshRenderer.material = reachGoalMaterial;
                break;
            case WinState.HalfWin:
                SetReward(reachGoalReward / 2f);
                floorMeshRenderer.material = halfWinMaterial;
                break;
            case WinState.Lose:
                SetReward(hitWallReward);
                floorMeshRenderer.material = hitWallMaterial;
                break;
            default:
                SetReward(0f);
                floorMeshRenderer.material = defaultMaterial;
                break;
        }

        //AddReward(GetAuxiliaryReward());
        //if (collision != null) Debug.Log("Relative velocity: " + Mathf.Abs(collision.relativeVelocity.y));
        //Debug.Log("Cummulative reward: " + GetCumulativeReward() + ", Divergence: " + divergence + ", Relative speed: " + collision?.relativeVelocity.y);
        EndEpisode();
    }

    public int GetDivergenceBinIndex(float divergence) {
        // [-infty, binSize, binSize*2, binSize*3, ..., binSize*bins, infty]
        int binIndex = 0;
        if(divergence > divergenceBinSize * (divergenceBins - 2)) {
            binIndex = divergenceBins - 1;
        }else {
            binIndex = (int)(divergence / divergenceBinSize);
        }
        return binIndex;
    }
    

    public float GetTrueDivergence() {
        float h = droneRigidbody.transform.localPosition.y - targetTransform.localPosition.y;
        if (Mathf.Abs(h) < 0.01) return 0f;
        return -droneRigidbody.velocity.y / h;
    }

    public cv2.Point2f[] Keypoints2Point2f(cv2.KeyPoint[] keypoints) {
        cv2.Point2f[] points = new cv2.Point2f[keypoints.Length];
        for (int i = 0; i < keypoints.Length; i++) {
            points[i] = keypoints[i].Pt;
        }
        return points;
    }


    public static bool AsBool(float value) {
        return Mathf.Approximately(Mathf.Min(value, 1), 1);
    }


}
