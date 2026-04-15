// ---------------------------------------------------------------------------
// Azure ML Workspace with all dependent resources (identity-based auth only)
// ---------------------------------------------------------------------------

@description('Base name for the workspace and dependent resources.')
param baseName string

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Enable managed virtual network isolation.')
param enableManagedVnet bool = false

@description('Tags applied to every resource.')
param tags object = {}

// ---------- Log Analytics ----------

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${baseName}-logs'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// ---------- Application Insights ----------

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${baseName}-ai'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ---------- Storage Account (identity-based, no key access) ----------

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: replace('${baseName}st', '-', '')
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowSharedKeyAccess: false
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

// ---------- Key Vault (RBAC mode, no access policies) ----------

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: '${baseName}-kv'
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

// ---------- Container Registry ----------

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: replace('${baseName}cr', '-', '')
  location: location
  tags: tags
  sku: {
    name: enableManagedVnet ? 'Premium' : 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

// ---------- Azure ML Workspace ----------

resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: '${baseName}-ws'
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: '${baseName} Workspace'
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: acr.id
    managedNetwork: enableManagedVnet
      ? {
          isolationMode: 'AllowInternetOutbound'
        }
      : null
  }
}

// ---------- Diagnostic Settings ----------

resource wsDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: '${baseName}-ws-diag'
  scope: workspace
  properties: {
    workspaceId: logAnalytics.id
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
  }
}

// ---------- Outputs ----------

output workspaceId string = workspace.id
output workspaceName string = workspace.name
output workspacePrincipalId string = workspace.identity.principalId
output storageAccountId string = storage.id
output keyVaultId string = keyVault.id
output acrId string = acr.id
output logAnalyticsId string = logAnalytics.id
output appInsightsId string = appInsights.id
