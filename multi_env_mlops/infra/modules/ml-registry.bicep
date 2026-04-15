// ---------------------------------------------------------------------------
// Shared Azure ML Registry for cross-workspace model promotion
// ---------------------------------------------------------------------------

@description('Name of the ML Registry.')
param registryName string

@description('Primary Azure region for the registry.')
param location string = resourceGroup().location

@description('Regions where registry assets can be replicated.')
param replicationLocations array = [location]

@description('Tags applied to the registry.')
param tags object = {}

resource mlRegistry 'Microsoft.MachineLearningServices/registries@2024-04-01' = {
  name: registryName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    regionDetails: [
      for loc in replicationLocations: {
        location: loc
        acrDetails: [
          {
            systemCreatedAcrAccount: {
              acrAccountSku: 'Premium'
            }
          }
        ]
        storageAccountDetails: [
          {
            systemCreatedStorageAccount: {
              storageAccountHnsEnabled: false
              storageAccountType: 'Standard_LRS'
            }
          }
        ]
      }
    ]
  }
}

output registryId string = mlRegistry.id
output registryName string = mlRegistry.name
