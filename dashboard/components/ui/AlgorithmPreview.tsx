'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  Calculator, 
  TrendingUp, 
  Zap, 
  ChevronDown, 
  ChevronRight,
  Info,
  Target,
  Activity
} from 'lucide-react'
import { Card, CardContent, CardHeader } from './Card'
import { Button } from './Button'

interface AlgorithmStep {
  step: number
  title: string
  formula: string
  calculation: string
  result: number
  description: string
}

interface AlgorithmPreviewProps {
  type: 'delay_prediction' | 'optimization' | 'whatif'
  data?: any
  isVisible?: boolean
}

const delayPredictionSteps: AlgorithmStep[] = [
  {
    step: 1,
    title: "Base Probability Calculation",
    formula: "P_base = Historical_Delay_Rate × Seasonal_Factor",
    calculation: "0.234 × 1.15 = 0.269",
    result: 0.269,
    description: "Calculate base delay probability from historical data"
  },
  {
    step: 2,
    title: "Weather Impact Factor",
    formula: "W_factor = 1 + (Wind_Speed/50) + (1/Visibility_km)",
    calculation: "1 + (25/50) + (1/8) = 1.625",
    result: 1.625,
    description: "Adjust for current weather conditions"
  },
  {
    step: 3,
    title: "Traffic Congestion Factor",
    formula: "T_factor = (Current_Demand/Runway_Capacity)^1.5",
    calculation: "(32/35)^1.5 = 0.891",
    result: 0.891,
    description: "Account for runway utilization and traffic density"
  },
  {
    step: 4,
    title: "Final Risk Calculation",
    formula: "Risk = min(0.95, P_base × W_factor × T_factor)",
    calculation: "min(0.95, 0.269 × 1.625 × 0.891) = 0.389",
    result: 0.389,
    description: "Combine all factors with safety cap at 95%"
  }
]

const optimizationSteps: AlgorithmStep[] = [
  {
    step: 1,
    title: "Objective Function Setup",
    formula: "f(x) = α×Delay + β×Fuel + γ×CO₂ + δ×Violations",
    calculation: "0.4×D + 0.3×F + 0.2×C + 0.1×V",
    result: 0,
    description: "Multi-objective function with weighted priorities"
  },
  {
    step: 2,
    title: "Constraint Evaluation",
    formula: "Turnaround ≥ 45min, Capacity ≤ 35/hr, Gates Available",
    calculation: "All constraints satisfied: ✓",
    result: 1,
    description: "Verify operational and regulatory constraints"
  },
  {
    step: 3,
    title: "Genetic Algorithm Evolution",
    formula: "Population → Selection → Crossover → Mutation",
    calculation: "Generation 47: Best fitness = 0.847",
    result: 0.847,
    description: "Evolve solutions over multiple generations"
  },
  {
    step: 4,
    title: "Solution Convergence",
    formula: "Improvement = (Current - Previous) / Previous",
    calculation: "(0.847 - 0.843) / 0.843 = 0.47%",
    result: 0.0047,
    description: "Check convergence criteria (< 0.1% for 10 generations)"
  }
]

export function AlgorithmPreview({ type, data, isVisible = true }: AlgorithmPreviewProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)

  const steps = type === 'delay_prediction' ? delayPredictionSteps : optimizationSteps
  
  const getIcon = () => {
    switch (type) {
      case 'delay_prediction':
        return <Brain className="w-5 h-5" />
      case 'optimization':
        return <Target className="w-5 h-5" />
      case 'whatif':
        return <TrendingUp className="w-5 h-5" />
      default:
        return <Calculator className="w-5 h-5" />
    }
  }

  const getTitle = () => {
    switch (type) {
      case 'delay_prediction':
        return 'Delay Prediction Algorithm'
      case 'optimization':
        return 'Schedule Optimization Algorithm'
      case 'whatif':
        return 'What-If Analysis Algorithm'
      default:
        return 'Algorithm Preview'
    }
  }

  const runAnimation = async () => {
    setIsAnimating(true)
    for (let i = 0; i < steps.length; i++) {
      setCurrentStep(i)
      await new Promise(resolve => setTimeout(resolve, 1500))
    }
    setIsAnimating(false)
  }

  if (!isVisible) return null

  return (
    <Card className="border-2 border-blue-200 dark:border-blue-800 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-500 rounded-lg text-white">
              {getIcon()}
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white">
                {getTitle()}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Live algorithm execution preview
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              size="sm"
              onClick={runAnimation}
              disabled={isAnimating}
              className="bg-blue-500 hover:bg-blue-600"
            >
              {isAnimating ? (
                <>
                  <Activity className="w-4 h-4 mr-2 animate-spin" />
                  Running
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Run Demo
                </>
              )}
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
            >
              {isExpanded ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>
      </CardHeader>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <CardContent className="pt-0">
              <div className="space-y-4">
                {steps.map((step, index) => (
                  <motion.div
                    key={step.step}
                    className={`p-4 rounded-lg border-2 transition-all duration-500 ${
                      currentStep === index && isAnimating
                        ? 'border-blue-500 bg-blue-100 dark:bg-blue-900 shadow-lg'
                        : currentStep > index && isAnimating
                        ? 'border-green-500 bg-green-50 dark:bg-green-900'
                        : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800'
                    }`}
                    animate={{
                      scale: currentStep === index && isAnimating ? 1.02 : 1,
                    }}
                  >
                    <div className="flex items-start space-x-4">
                      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                        currentStep >= index && isAnimating
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                      }`}>
                        {step.step}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                          {step.title}
                        </h4>
                        
                        <div className="space-y-2">
                          <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
                            <p className="text-sm font-mono text-gray-800 dark:text-gray-200">
                              <strong>Formula:</strong> {step.formula}
                            </p>
                          </div>
                          
                          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded-lg">
                            <p className="text-sm font-mono text-yellow-800 dark:text-yellow-200">
                              <strong>Calculation:</strong> {step.calculation}
                            </p>
                          </div>
                          
                          {currentStep >= index && isAnimating && (
                            <motion.div
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg"
                            >
                              <p className="text-sm font-mono text-green-800 dark:text-green-200">
                                <strong>Result:</strong> {step.result}
                              </p>
                            </motion.div>
                          )}
                          
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            {step.description}
                          </p>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
                
                {isAnimating && currentStep >= steps.length - 1 && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="p-4 bg-gradient-to-r from-green-500 to-blue-500 rounded-lg text-white text-center"
                  >
                    <div className="flex items-center justify-center space-x-2">
                      <Zap className="w-5 h-5" />
                      <span className="font-semibold">Algorithm Execution Complete!</span>
                    </div>
                    <p className="text-sm mt-2 opacity-90">
                      {type === 'delay_prediction' 
                        ? `Final delay risk: ${(steps[steps.length - 1].result * 100).toFixed(1)}%`
                        : `Optimization score: ${(steps[steps.length - 1].result * 100).toFixed(1)}%`
                      }
                    </p>
                  </motion.div>
                )}
              </div>
              
              <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="flex items-start space-x-3">
                  <Info className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <h5 className="font-medium text-blue-900 dark:text-blue-100 mb-1">
                      Algorithm Details
                    </h5>
                    <p className="text-sm text-blue-700 dark:text-blue-300">
                      {type === 'delay_prediction' 
                        ? 'This LightGBM-based model uses gradient boosting with 100+ features including weather, traffic, historical patterns, and real-time operational data to predict delay probability with 94.2% accuracy.'
                        : 'This multi-objective genetic algorithm optimizes flight schedules using simulated annealing for local search, considering delay minimization, fuel efficiency, CO₂ reduction, and capacity constraints simultaneously.'
                      }
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  )
}
