import { useCallback, useEffect, useState } from 'react'
import type { AnalyseResponse } from '../types'

const STORAGE_KEY = 'aferidor.history.v1'

export interface HistoryEntry {
  id: string
  preview: string // a short, truncated snippet of the analysed text (for the list)
  result: AnalyseResponse // the full response, so re-viewing never re-sends the text
  at: number // epoch milliseconds
}

const MAX_ENTRIES = 20
const PREVIEW_CHARS = 160

function load(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : []
  } catch {
    return [] // corrupt/unavailable storage → behave as empty rather than crash
  }
}

function persist(entries: HistoryEntry[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
  } catch {
  }
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID()
  return `${Date.now()}-${Math.floor(Math.random() * 1e6)}`
}

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>([])

  useEffect(() => {
    setEntries(load())
  }, [])

  const add = useCallback((text: string, result: AnalyseResponse) => {
    setEntries((prev) => {
      const entry: HistoryEntry = {
        id: makeId(),
        preview: text.slice(0, PREVIEW_CHARS),
        result,
        at: Date.now(),
      }
      const next = [entry, ...prev].slice(0, MAX_ENTRIES) // newest first; cap by count
      persist(next)
      return next
    })
  }, [])

  const remove = useCallback((id: string) => {
    setEntries((prev) => {
      const next = prev.filter((e) => e.id !== id)
      persist(next)
      return next
    })
  }, [])

  const clear = useCallback(() => {
    setEntries([])
    persist([])
  }, [])

  return { entries, add, remove, clear }
}
