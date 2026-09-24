"use client"

import React, { useState } from "react"
import { 
  Clapperboard, 
  Film, 
  ShieldAlert, 
  PieChart, 
  Activity, 
  Database, 
  Cpu, 
  ChevronRight,
  Sparkles
} from "lucide-react"

import TextRoll from "@/components/ui/text-roll"
import { cn } from "@/lib/utils"

export interface NavItem {
  id: string
  name: string
  description: string
  icon: React.ComponentType<{ className?: string }>
  badge?: string
}

const NAV_ITEMS: NavItem[] = [
  {
    id: "recommendations",
    name: "Recommendation Engine",
    description: "TF-IDF + Cosine Similarity matching",
    icon: Clapperboard,
    badge: "NLP",
  },
  {
    id: "format-classifier",
    name: "Content Type Predictor",
    description: "Movie vs TV Show binary classification",
    icon: Film,
    badge: "100%",
  },
  {
    id: "rating-classifier",
    name: "Rating Classifier",
    description: "Multi-class audience maturity scoring",
    icon: ShieldAlert,
    badge: "9 Classes",
  },
  {
    id: "segmentation",
    name: "Content Segmentation",
    description: "K-Means unsupervised clustering & PCA",
    icon: PieChart,
    badge: "K=5",
  },
]

export default function SidebarNav({
  activeId = "recommendations",
  onSelect,
  className,
}: {
  activeId?: string
  onSelect?: (id: string) => void
  className?: string
}) {
  const [selected, setSelected] = useState<string>(activeId)

  const handleSelect = (id: string) => {
    setSelected(id)
    onSelect?.(id)
  }

  return (
    <aside
      className={cn(
        "flex h-full min-h-screen w-80 flex-col justify-between border-r border-zinc-800/80 bg-[#0A0D14] p-5 text-zinc-100 antialiased",
        className
      )}
    >
      {/* Top Section: Brand & Navigation */}
      <div className="flex flex-col gap-6">
        {/* Brand Header */}
        <div className="flex items-center justify-between border-b border-zinc-800/60 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#E50914] shadow-md shadow-red-950/40">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-sm font-extrabold tracking-wider text-white">
                NETFLIX <span className="text-[#E50914]">ML</span>
              </h2>
              <p className="text-[11px] font-medium text-zinc-400">Analytics Engine</p>
            </div>
          </div>
          <span className="rounded-full border border-red-500/20 bg-red-500/10 px-2 py-0.5 text-[10px] font-semibold text-red-400">
            v1.0
          </span>
        </div>

        {/* Navigation Section */}
        <div>
          <div className="mb-3 flex items-center justify-between">
            <span className="text-[11px] font-bold tracking-wider text-zinc-400 uppercase">
              Navigation Modules
            </span>
          </div>

          <nav className="flex flex-col gap-1.5">
            {NAV_ITEMS.map((item) => {
              const isActive = selected === item.id
              const Icon = item.icon

              return (
                <button
                  key={item.id}
                  onClick={() => handleSelect(item.id)}
                  className={cn(
                    "group relative flex w-full items-center justify-between rounded-xl p-3 text-left transition-all duration-200",
                    isActive
                      ? "border border-red-500/30 bg-gradient-to-r from-red-950/30 via-zinc-900/60 to-zinc-900/40 shadow-lg shadow-red-950/20"
                      : "border border-transparent hover:border-zinc-800 hover:bg-zinc-900/50"
                  )}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={cn(
                        "flex h-9 w-9 items-center justify-center rounded-lg transition-colors",
                        isActive
                          ? "bg-[#E50914] text-white shadow-md shadow-red-950/40"
                          : "bg-zinc-900 text-zinc-400 group-hover:bg-zinc-800 group-hover:text-zinc-200"
                      )}
                    >
                      <Icon className="h-4 w-4" />
                    </div>

                    <div className="flex flex-col">
                      {/* TextRoll Integration */}
                      <TextRoll
                        className={cn(
                          "text-sm font-bold tracking-tight transition-colors",
                          isActive
                            ? "text-white"
                            : "text-zinc-300 group-hover:text-white"
                        )}
                      >
                        {item.name}
                      </TextRoll>
                      <span className="text-[11px] text-zinc-500 group-hover:text-zinc-400">
                        {item.description}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {item.badge && (
                      <span
                        className={cn(
                          "rounded-md px-1.5 py-0.5 text-[10px] font-semibold transition-colors",
                          isActive
                            ? "bg-red-500/20 text-red-300"
                            : "bg-zinc-800/80 text-zinc-400 group-hover:bg-zinc-800 group-hover:text-zinc-300"
                        )}
                      >
                        {item.badge}
                      </span>
                    )}
                    <ChevronRight
                      className={cn(
                        "h-4 w-4 transition-transform duration-200",
                        isActive
                          ? "translate-x-0.5 text-[#E50914]"
                          : "text-zinc-600 opacity-0 group-hover:opacity-100"
                      )}
                    />
                  </div>
                </button>
              )
            })}
          </nav>
        </div>
      </div>

      {/* Bottom Section: Telemetry, Dataset, Architecture */}
      <div className="mt-8 flex flex-col gap-3 border-t border-zinc-800/60 pt-4">
        {/* System Status Pill */}
        <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-3.5 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500"></span>
              </span>
              <span className="text-xs font-semibold text-zinc-200">System Status</span>
            </div>
            <span className="text-[10px] font-medium text-emerald-400">Online & Ready</span>
          </div>

          <div className="mt-2.5 flex items-center justify-between border-t border-zinc-800/60 pt-2 text-[11px] text-zinc-400">
            <span className="flex items-center gap-1.5">
              <Database className="h-3 w-3 text-zinc-500" />
              Dataset
            </span>
            <span className="font-mono text-zinc-300">8,790 records</span>
          </div>
        </div>

        {/* Architecture Pill */}
        <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/20 p-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300">
            <Cpu className="h-3.5 w-3.5 text-red-500" />
            <span>Architecture</span>
          </div>
          <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
            Scikit-Learn • Random Forest • TF-IDF • K-Means • Streamlit
          </p>
        </div>
      </div>
    </aside>
  )
}
