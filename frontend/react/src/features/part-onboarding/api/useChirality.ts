/** The part's left/right map status (display only); polls while the server computes it. */
import { useQuery } from '@tanstack/react-query'
import { onboardingApi, onboardingKeys } from './onboardingApi'

export function useChirality(partId: string, recipeId: string | null | undefined) {
  return useQuery({
    queryKey: [...onboardingKeys.chirality(partId), recipeId ?? null],
    queryFn: () => onboardingApi.getChirality(partId),
    enabled: !!partId && !!recipeId,
    retry: false,
    refetchOnWindowFocus: false,
    refetchInterval: query => (query.state.data?.status === 'PENDING' ? 3000 : false),
  })
}
