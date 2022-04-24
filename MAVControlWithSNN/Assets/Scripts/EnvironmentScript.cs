using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentScript : MonoBehaviour
{

    [Header("Parameters")]
    public float density = 0.01f; // nb_markers / unit^3
    public Vector2 markersMinMaxX = new Vector2(-15f, 15f);
    public Vector2 markersMinMaxY = new Vector2(0f, 100f);
    public Vector2 markersMinMaxZ = new Vector2(-5f, 5f);

    [Header("Materials")]
    public Material[] materials;

    [Header("References")]
    public GameObject ceilling;
    [SerializeField] private MeshRenderer groundRenderer;
    [SerializeField] private GameObject markersRoot;
    public GameObject[] markerPrefabs;
    private List<GameObject> markers = new List<GameObject>();
    [SerializeField] private Wall[] walls;


    // Start is called before the first frame update
    void Start()
    {
        walls = GetComponentsInChildren<Wall>();
        //CreateRandomMarkers();
        InitializeWalls();
    }

    // Update is called once per frame
    void Update()
    {
        
    }


    public void InitializeWalls() {
        foreach (Wall wall in walls) {
            wall.CreateRandomMarkers();
        }
    }


    public void Randomize() {
        Destroy(groundRenderer.material);
        groundRenderer.material = materials[Random.Range(0, materials.Length)];
        groundRenderer.material.mainTextureScale = new Vector2(Random.Range(1, 20), Random.Range(1, 20));

        markersMinMaxY.y = ceilling.transform.localPosition.y;

        foreach (Wall wall in walls) {
            wall.Randomize();
        }
    }


    private void EnableUsedMarkers() {
        int nb_markers = ComputeNbMarkers();
        for (int i = 0; i < markers.Count; i++) {
            if (i < nb_markers) {
                markers[i].SetActive(true);
                RandomizeMarker(markers[i]);
            }
            else {
                markers[i].SetActive(false);
            }
        }
    }


    private void DestroyMarkers() {
        foreach (GameObject marker in markers) {
            Destroy(marker);
        }
        markers.Clear();
    }


    private void CreateRandomMarkers() {
        int nb_markers = ComputeNbMarkers();
        for (int i = markers.Count; i < nb_markers; i++) {
            GameObject marker = GameObject.Instantiate(markerPrefabs[Random.Range(0, markerPrefabs.Length)], markersRoot.transform);
            markers.Add(marker);
            RandomizeMarker(marker);
        }
    }

    private void RandomizeMarker(GameObject marker) {
        float x = Random.Range(markersMinMaxX.x, markersMinMaxX.y);
        float y = Random.Range(markersMinMaxY.x, markersMinMaxY.y);
        //float y = Random.Range(markersMinMaxY.x, ceilling.transform.localPosition.y);
        float z = Random.Range(markersMinMaxZ.x, markersMinMaxZ.y);
        marker.transform.localPosition = new Vector3(x, y, z);

        Destroy(marker.GetComponent<MeshRenderer>().material);
        marker.GetComponent<MeshRenderer>().material = materials[Random.Range(0, materials.Length)];
        marker.transform.localScale = new Vector3(Random.Range(0.1f, 1.5f), Random.Range(0.1f, 1.5f), Random.Range(0.1f, 1.5f));
    }

    public void RandomizeMarkers() {
        foreach (GameObject marker in markers) {
            RandomizeMarker(marker);
        }
    }


    private int ComputeNbMarkers() {
        float dx = Mathf.Max(1f, Mathf.Abs(markersMinMaxX.y - markersMinMaxX.x));
        float dy = Mathf.Max(1f, Mathf.Abs(markersMinMaxY.y - markersMinMaxY.x));
        float dz = Mathf.Max(1f, Mathf.Abs(markersMinMaxZ.y - markersMinMaxZ.x));
        float volume = dx * dy * dz;
        return (int)(density * volume);
    }


}
