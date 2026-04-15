// ---------------------------------------------------------------------------
// Multi-Environment MLOps — Main Bicep Orchestrator
//
// Deploys:
//   - Azure ML Workspace + dependencies (Storage, Key Vault, ACR, App Insights, Log Analytics)
//   - Compute cluster
//   - RBAC role assignments (identity-based, no keys)
//   - Optionally: managed VNet isolation
//
// The shared ML Registry is deployed separately via shared.bicep into its own
// resource group. Pass its resource ID here so workspace RBAC can reference it.
// ---------------------------------------------------------------------------

targetScope = 'resourceGroup'

@description('Environment name used as a prefix (e.g. dev, test, prod).')
param environment string

@description('Project short name.')
param projectName string = 'readmit'

@description('Azure region.')
param location string = resourceGroup().location

@description('Enable managed virtual network for the workspace.')
param enableManagedVnet bool = false

@description('Resource ID of the shared ML Registry (deployed via shared.bicep). Leave empty to skip registry RBAC.')
param mlRegistryId string = ''

@description('VM size for compute cluster.')
param computeVmSize string = 'Standard_DS3_v2'

@description('Max nodes for compute cluster.')
param computeMaxNodes int = 4

@description('Principal ID of the user to grant Contributor on the resource group.')
param userPrincipalId string = ''

// ---------- Derived names ----------

var baseName = '${projectName}-${environment}'

var tags = {
  project: projectName
  environment: environment
  managedBy: 'bicep'
}

// ---------- ML Workspace + dependencies ----------

module workspace 'modules/ml-workspace.bicep' = {
  name: 'workspace-${environment}'
  params: {
    baseName: baseName
    location: location
    enableManagedVnet: enableManagedVnet
    tags: tags
  }
}

// ---------- Compute ----------

module compute 'modules/ml-compute.bicep' = {
  name: 'compute-${environment}'
  params: {
    workspaceId: workspace.outputs.workspaceId
    clusterName: 'cpu-cluster'
    vmSize: computeVmSize
    minNodeCount: 0
    maxNodeCount: computeMaxNodes
    location: location
    tags: tags
  }
}

// ---------- RBAC ----------

module roles 'modules/role-assignments.bicep' = {
  name: 'roles-${environment}'
  params: {
    workspacePrincipalId: workspace.outputs.workspacePrincipalId
    computePrincipalId: compute.outputs.computePrincipalId
    storageAccountId: workspace.outputs.storageAccountId
    keyVaultId: workspace.outputs.keyVaultId
    acrId: workspace.outputs.acrId
    userPrincipalId: userPrincipalId
    workspaceId: workspace.outputs.workspaceId
  }
}

// ---------- Registry RBAC (cross-resource-group) ----------

// Extract the resource group name from the registry resource ID
var registryRgName = !empty(mlRegistryId) ? split(mlRegistryId, '/')[4] : resourceGroup().name

module registryRbac 'modules/registry-role.bicep' = if (!empty(mlRegistryId)) {
  name: 'registry-rbac-${environment}'
  scope: resourceGroup(registryRgName)
  params: {
    mlRegistryId: mlRegistryId
    principalId: workspace.outputs.workspacePrincipalId
  }
}

// ---------- Outputs ----------

output workspaceName string = workspace.outputs.workspaceName
output workspaceId string = workspace.outputs.workspaceId
output computeName string = compute.outputs.computeName
