/** Optional agent-facing navigation for browsers that implement WebMCP. */
type DemoTool = {
  name: string
  title: string
  description: string
  inputSchema: object
  annotations: { readOnlyHint: boolean; untrustedContentHint: boolean }
  execute: (input: unknown) => Promise<unknown>
}

const modelContext = (document as Document & {
  modelContext?: { registerTool: (tool: DemoTool) => void | Promise<void> }
}).modelContext

if (modelContext?.registerTool) {
  const parts = fetch('/vendor-corpus/manifest.json').then(async response => {
    if (!response.ok) throw new Error('Demo parts unavailable')
    const data = await response.json() as { parts: { id: string; part_number: string; candidate_count: number }[] }
    return data.parts
  })

  void Promise.resolve(modelContext.registerTool({
    name: 'list_vendor_demo_parts',
    title: 'List demo CAD parts',
    description: 'List the five CAD parts available in this vendor bend-tagging demo.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true, untrustedContentHint: false },
    execute: async () => (await parts).map(({ id, part_number, candidate_count }) => ({ id, part_number, candidate_count })),
  })).catch(() => {})

  void Promise.resolve(modelContext.registerTool({
    name: 'select_vendor_demo_part',
    title: 'Select demo CAD part',
    description: 'Open one of the five available CAD parts in the visible bend-tagging workspace.',
    inputSchema: {
      type: 'object',
      properties: { id: { type: 'string' } },
      required: ['id'],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false, untrustedContentHint: false },
    execute: async input => {
      const id = (input as { id?: unknown } | null)?.id
      if (typeof id !== 'string' || !(await parts).some(part => part.id === id)) throw new Error('Unknown demo part')
      window.location.hash = `/parts/vendor-setup/${id}`
      return { selected_part_id: id }
    },
  })).catch(() => {})
}
