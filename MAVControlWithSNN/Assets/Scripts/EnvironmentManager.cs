using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;




public class EnvironmentManager : MonoBehaviour
{
    private EnvironmentParameters environmentParameters;
    [SerializeField] private GameObject environmentPrefab;
    private Vector3 envPrefabSize;
    [SerializeField] private int batchSize = 32;
    [SerializeField] private float padSize = 1f;
    private List<EnvironmentScript> environments;

    // Start is called before the first frame update
    void Start()
    {
        InstantiateEnvironments();
    }

    // Update is called once per frame
    void Update()
    {
        
    }


    public List<EnvironmentScript> GetEnvironments() {
        return environments;
    }


    private void InitializeReferences() {
        environmentParameters = Academy.Instance.EnvironmentParameters;
        batchSize = (int)environmentParameters.GetWithDefault("batchSize", (float)batchSize);  // batchSize is the number of env or agents
        envPrefabSize = environmentPrefab.GetComponentInChildren<Renderer>().bounds.size;
    }



    public void InstantiateEnvironments() {
        int side0 = (int)Mathf.Ceil(Mathf.Sqrt((float)batchSize));
        int side1 = batchSize / side0;

        InitializeReferences();
        environments = new List<EnvironmentScript>((EnvironmentScript[])GameObject.FindObjectsOfType(typeof(EnvironmentScript)));
        if(environments.Count > batchSize) {
            for (int i = environments.Count-1; i >= batchSize-1; i--) {
                DestroyImmediate(environments[i].gameObject);
            }
        }
        environments = new List<EnvironmentScript>((EnvironmentScript[])GameObject.FindObjectsOfType(typeof(EnvironmentScript)));
        for (int i = 0; i < environments.Count; i++) {
            float x = -(i % side0) * (envPrefabSize.x + padSize);
            float z = -(i / side0) * (envPrefabSize.x + padSize);
            environments[i].transform.position = new Vector3(x, 0f, z);
        }
        for (int i = environments.Count; i < batchSize; i++) {
            float x = -(i % side0) * (envPrefabSize.x + padSize);
            float z = -(i / side0) * (envPrefabSize.x + padSize);
            GameObject env = Instantiate(environmentPrefab, new Vector3(x, 0f, z), Quaternion.identity);
            environments.Add(env.GetComponent<EnvironmentScript>());
        }
    }



}
