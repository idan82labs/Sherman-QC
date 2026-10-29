import { api } from '../../services/api'
import type { AssetInventoryResponse, HumanBendLabels } from './types'

const base = '/training/bend-labeling'

export const bendLabelingApi = {
  listAssets: async (): Promise<AssetInventoryResponse> => {
    const response = await api.get(`${base}/assets`)
    return response.data
  },
  labels: async (assetId: string): Promise<HumanBendLabels> => {
    const response = await api.get(`${base}/assets/${assetId}/labels`)
    return response.data
  },
  save: async (assetId: string, labels: HumanBendLabels): Promise<HumanBendLabels> => {
    const response = await api.put(`${base}/assets/${assetId}/labels`, labels)
    return response.data
  },
  assetUrl: (assetId: string) => `/api/training/bend-labeling/assets/${assetId}/file`,
}
