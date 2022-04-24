using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Wall : MonoBehaviour
{
    public float density = 0.01f; // nb_markers / unit^3
    private List<GameObject> markers = new List<GameObject>();
    [SerializeField] private EnvironmentScript m_environmentScript;
    [SerializeField] private BoxCollider m_collider;


    private void Awake() {
        m_environmentScript = GetComponentInParent<EnvironmentScript>();
        m_collider = GetComponent<BoxCollider>();
    }


    // Start is called before the first frame update
    void Start()
    {
        m_environmentScript = GetComponentInParent<EnvironmentScript>();
        m_collider = GetComponent<BoxCollider>();
    }



    // Update is called once per frame
    void Update()
    {
        
    }


    public void SetReferences() {

    }


    public void Randomize() {
        if(m_collider == null) m_collider = GetComponent<BoxCollider>();

        CreateRandomMarkers();
        EnableUsedMarkers();
    }




    public void EnableUsedMarkers() {
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


    public void DestroyMarkers() {
        foreach (GameObject marker in markers) {
            Destroy(marker);
        }
        markers.Clear();
    }


    public void CreateRandomMarkers() {
        int nb_markers = ComputeNbMarkers();
        for (int i = markers.Count; i < nb_markers; i++) {
            GameObject rn_prefab = m_environmentScript.markerPrefabs[Random.Range(0, m_environmentScript.markerPrefabs.Length)];
            GameObject marker = GameObject.Instantiate(rn_prefab, transform);
            markers.Add(marker);
            RandomizeMarker(marker);
        }
    }


    public void RandomizeMarker(GameObject marker) {
        float x = Random.Range(-m_collider.bounds.extents.x, m_collider.bounds.extents.x);
        float y = Random.Range(-m_collider.bounds.extents.y, m_environmentScript.ceilling.transform.localPosition.y);
        float z = Random.Range(-m_collider.bounds.extents.z, m_collider.bounds.extents.z);
        //marker.transform.localPosition = new Vector3(x, 0, z);
        Vector3 new_position = GetRandomPointInsideCollider(m_collider);
        Vector3 new_altitude = transform.TransformPoint(new Vector3(0f, m_collider.center.y - m_collider.size.y / 2f, 0f));
        new_position.y = Random.Range(new_altitude.y, m_environmentScript.ceilling.transform.position.y);
        //new_position.y = m_environmentScript.ceilling.transform.position.y;
        marker.transform.position = new_position;


        Destroy(marker.GetComponent<MeshRenderer>().material);
        marker.GetComponent<MeshRenderer>().material = m_environmentScript.materials[Random.Range(0, m_environmentScript.materials.Length)];

        float sx = Random.Range(0.1f, 1f) / transform.localScale.x;
        float sy = Random.Range(0.1f, 1f) / transform.localScale.y;
        float sz = Random.Range(0.1f, 1f) / transform.localScale.z;

        marker.transform.localScale = new Vector3(sx, sy, sz);
    }

    public void RandomizeMarkers() {
        foreach (GameObject marker in markers) {
            RandomizeMarker(marker);
        }
    }


    public int ComputeNbMarkers() {
        float dx = Mathf.Max(1f, Mathf.Abs(m_collider.size.x * transform.localScale.x));
        float dy = Mathf.Max(1f, Mathf.Abs(m_environmentScript.ceilling.transform.localPosition.y));
        float dz = Mathf.Max(1f, Mathf.Abs(m_collider.size.z * transform.localScale.z));
        Vector3 size = new Vector3(
            Mathf.Max(1f, Mathf.Abs(m_collider.size.x * transform.localScale.x)),
            Mathf.Max(1f, Mathf.Abs(m_environmentScript.ceilling.transform.localPosition.y - (m_collider.size.y/2f + m_collider.center.y))),
            Mathf.Max(1f, Mathf.Abs(m_collider.size.z * transform.localScale.z))
        );
        float volume = size.x * size.y * size.z;
        //Debug.Log("size: " + size + ", volume: " + volume + ", nb_markers: " + (int)(density * volume));
        return (int)(density * volume);
    }


    public Vector3 GetRandomPointInsideCollider(BoxCollider boxCollider) {
        //Vector3 extents = Vector3.Scale(boxCollider.transform.localScale, boxCollider.size) / 2f;
        Vector3 extents = boxCollider.size / 2f;
        Vector3 point = new Vector3(
            Random.Range(-extents.x, extents.x),
            Random.Range(-extents.y, extents.y),
            Random.Range(-extents.z, extents.z)
        ) + boxCollider.center;
        return boxCollider.transform.TransformPoint(point);
    }


}
