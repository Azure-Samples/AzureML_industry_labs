// ---------------------------------------------------------------------------
// Azure ML Compute Cluster (declarative, not created at pipeline runtime)
// ---------------------------------------------------------------------------

@description('Resource ID of the Azure ML Workspace.')
param workspaceId string

@description('Name of the compute cluster.')
param clusterName string = 'cpu-cluster'

@description('VM size for the cluster nodes.')
param vmSize string = 'Standard_DS3_v2'

@description('Minimum node count (0 = scale to zero).')
param minNodeCount int = 0

@description('Maximum node count.')
param maxNodeCount int = 4

@description('Azure region.')
param location string = resourceGroup().location

@description('Tags.')
param tags object = {}

// Extract workspace name from its resource ID
var workspaceName = last(split(workspaceId, '/'))

resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' existing = {
  name: workspaceName
}

resource compute 'Microsoft.MachineLearningServices/workspaces/computes@2024-04-01' = {
  parent: workspace
  name: clusterName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: vmSize
      scaleSettings: {
        minNodeCount: minNodeCount
        maxNodeCount: maxNodeCount
        nodeIdleTimeBeforeScaleDown: 'PT5M'
      }
    }
  }
}

output computeName string = compute.name
output computePrincipalId string = compute.identity.principalId
