using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;



[CustomEditor(typeof(EnvironmentManager))]
public class EnvironmentManagerEditor : Editor {
    public override void OnInspectorGUI() {
        EnvironmentManager myTarget = (EnvironmentManager)target;

        DrawDefaultInspector();

        if (GUILayout.Button("Instantiate Environments")) {
            myTarget.InstantiateEnvironments();
        }
    }
}