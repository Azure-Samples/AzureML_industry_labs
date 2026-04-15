// ---------------------------------------------------------------------------
// Cross-resource-group role assignment: Workspace MI → ML Registry
//
// This module is deployed to the registry's resource group via
// scope: resourceGroup(...) from main.bicep so the 'existing' lookup
// resolves correctly.
// ---------------------------------------------------------------------------

@description('Full resource ID of the ML Registry.')
param mlRegistryId string

@description('Principal ID to grant AzureML Registry User.')
param principalId string

var registryName = last(split(mlRegistryId, '/'))
var azureMLRegistryUser = '1823dd4f-9b8c-4ab6-ab4e-7397a3684615'

resource registryRef 'Microsoft.MachineLearningServices/registries@2024-04-01' existing = {
  name: registryName
}

resource registryRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(mlRegistryId, principalId, azureMLRegistryUser)
  scope: registryRef
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', azureMLRegistryUser)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}
