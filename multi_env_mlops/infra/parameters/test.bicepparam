using '../main.bicep'

param environment = 'test'
param projectName = 'readmit'
param location = 'swedencentral'
param enableManagedVnet = true
param mlRegistryId = ''
param computeVmSize = 'Standard_DS3_v2'
param computeMaxNodes = 2
param userPrincipalId = ''
