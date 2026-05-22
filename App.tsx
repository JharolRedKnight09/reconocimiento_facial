import { Camera, FileCode2, Shield, Terminal, LayoutTemplate } from 'lucide-react';
import React from 'react';

export default function App() {
  const codeSnippet = `import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis

# ... Código base con Interfaz Gráfica Moderna en reconocimiento_facial.py
# Fases implementadas:
# 1. UI Moderna Dark Mode (CustomTkinter)
# 2. Enrolamiento y Registro Visual
# 3. Flujo Automatico Biometrico Ultra Rapido
# 4. Eliminacion de Usuarios desde el Menu`;

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-gray-200 font-sans p-6 sm:p-12">
      <div className="max-w-4xl mx-auto space-y-12">
        {/* Header */}
        <header className="space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-500/10 text-blue-400 rounded-full text-sm font-medium border border-blue-500/20">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
            </span>
            ¡Nueva Interfaz Grafica Disponible!
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-white">
            Sistema de Acceso Biometrico - GUI Premium
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl leading-relaxed">
            Se ha re-diseñado completamente el script <code className="text-blue-400 bg-blue-400/10 px-2 py-0.5 rounded">reconocimiento_facial.py</code>. Ahora utiliza <b>CustomTkinter</b> para una interfaz grafica oscura, moderna y fluida sin necesidad de consola.
          </p>
        </header>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FeatureCard
            icon={<LayoutTemplate className="w-5 h-5 text-purple-400" />}
            title="Diseño Dark UI"
            description="Botones, paneles laterales estilo dashboard y renderizado en vivo del video dentro de la misma ventana."
          />
          <FeatureCard
            icon={<Camera className="w-5 h-5 text-emerald-400" />}
            title="Video Ultra Fluido (Threads)"
            description="Cero congelamientos. Se usa Multi-Threading (hilos) para analizar el rostro sin frenar la camara."
          />
          <FeatureCard
            icon={<Shield className="w-5 h-5 text-indigo-400" />}
            title="Desbloqueo Automatico Continuo"
            description="Ya no es necesario presionar un botón. El sistema se desbloquea en 1 segundo cuando verifica tu cara y sigue escaneando."
          />
          <FeatureCard
            icon={<Terminal className="w-5 h-5 text-orange-400" />}
            title="Gestion de BD Integrada"
            description="Ventanas modales limpias para insertar, enrolar o borrar el acceso a usuarios con solo un par de clics."
          />
        </div>

        {/* Installation & Execution */}
        <div className="bg-[#141414] rounded-xl border border-white/10 overflow-hidden shadow-2xl">
          <div className="px-6 py-4 border-b border-white/5 bg-black/40 flex items-center justify-between">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-400" />
              Pasos para la Nueva Interfaz
            </h3>
          </div>
          <div className="p-6 space-y-6">
            <div className="space-y-2">
               <p className="text-sm font-medium text-gray-400">1. Instala el motor grafico para Python (CustomTkinter y Pillow):</p>
               <div className="bg-black/50 p-3 rounded-lg font-mono text-sm text-green-400 flex items-center justify-between group">
                 <span>py -m pip install customtkinter pillow</span>
               </div>
            </div>

            <div className="space-y-2">
               <p className="text-sm font-medium text-gray-400">2. Descarga el codigo actualizado de la plataforma:</p>
               <div className="bg-black/50 p-4 rounded-lg font-mono text-sm text-gray-300 whitespace-pre-wrap overflow-x-auto">
                 {codeSnippet}
               </div>
            </div>

            <div className="space-y-2">
               <p className="text-sm font-medium text-gray-400">3. Inicia la aplicacion visual:</p>
               <div className="bg-black/50 p-3 rounded-lg font-mono text-sm text-blue-400 flex items-center justify-between">
                 <span>py src/reconocimiento_facial.py</span>
               </div>
            </div>
            
          </div>
        </div>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="p-5 rounded-xl border border-white/5 bg-[#141414] hover:bg-[#1A1A1A] transition-colors">
      <div className="flex items-center gap-3 mb-3">
        <div className="p-2 bg-white/5 rounded-lg">
          {icon}
        </div>
        <h3 className="font-semibold text-white">{title}</h3>
      </div>
      <p className="text-sm text-gray-400 leading-relaxed">
        {description}
      </p>
    </div>
  );
}

