// ---------------------------------------------------------------------------
// Shared Infrastructure — ML Registry for cross-environment promotion
//
// Deployed to a dedicated resource group (e.g. rg-readmit-shared) so the
// registry lifecycle is independent of any single environment.
// ---------------------------------------------------------------------------

targetScope = 'resourceGroup'

@description('Project short name.')
param projectName string = 'readmit'

@description('Azure region.')
param location string = resourceGroup().location

var registryName = '${projectName}-registry'

var tags = {
  project: projectName
  environment: 'shared'
  managedBy: 'bicep'
}

module registry 'modules/ml-registry.bicep' = {
  name: 'registry'
  params: {
    registryName: registryName
    location: location
    replicationLocations: [location]
    tags: tags
  }
}

output registryId string = registry.outputs.registryId
output registryName string = registry.outputs.registryName
