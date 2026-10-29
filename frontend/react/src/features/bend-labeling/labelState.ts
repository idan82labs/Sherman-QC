import type { BendLineLabel, HumanBendLabels, Point3 } from './types'

export function nextBendId(bends: BendLineLabel[]): string {
  const used = new Set(bends.map((bend) => bend.bend_id))
  let index = 1
  while (used.has(`B${index}`)) index += 1
  return `B${index}`
}

export function addBend(labels: HumanBendLabels, start: Point3, end: Point3): HumanBendLabels {
  if (start.every((value, index) => value === end[index])) {
    throw new Error('Start and end must be different points.')
  }
  return {
    ...labels,
    review_status: 'in_progress',
    no_bends_present: false,
    bends: [...labels.bends, {
      bend_id: nextBendId(labels.bends),
      line_start: start,
      line_end: end,
      confidence: 'certain',
      notes: '',
    }],
  }
}

export function replaceEndpoint(
  labels: HumanBendLabels,
  bendId: string,
  endpoint: 'start' | 'end',
  point: Point3,
): HumanBendLabels {
  return {
    ...labels,
    review_status: 'in_progress',
    bends: labels.bends.map((bend) => bend.bend_id !== bendId ? bend : {
      ...bend,
      [endpoint === 'start' ? 'line_start' : 'line_end']: point,
    }),
  }
}

export function removeBend(labels: HumanBendLabels, bendId: string): HumanBendLabels {
  return {
    ...labels,
    review_status: 'in_progress',
    bends: labels.bends.filter((bend) => bend.bend_id !== bendId),
  }
}
