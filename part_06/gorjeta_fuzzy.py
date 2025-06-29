import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GorjetaFuzzy:
    def __init__(self):
        self.comida = ctrl.Antecedent(np.linspace(0, 1, 101), 'comida')
        self.servico = ctrl.Antecedent(np.linspace(0, 1, 101), 'servico')
        self.gorjeta = ctrl.Consequent(np.linspace(0, 20, 101), 'gorjeta')
        self._definir_pertinencias()
        self._definir_regras()
        self.sistema = ctrl.ControlSystem(self.regras)
        self.simulador = ctrl.ControlSystemSimulation(self.sistema)
    
    def _definir_pertinencias(self):
        self.comida['ruim'] = fuzz.trimf(self.comida.universe, [0.0, 0.0, 0.5])
        self.comida['boa'] = fuzz.trimf(self.comida.universe, [0.0, 0.5, 1.0])
        self.comida['saborosa'] = fuzz.trimf(self.comida.universe, [0.5, 1.0, 1.0])

        self.servico['ruim'] = fuzz.trimf(self.servico.universe, [0.0, 0.0, 0.5])
        self.servico['aceitavel'] = fuzz.trimf(self.servico.universe, [0.0, 0.5, 1.0])
        self.servico['otima'] = fuzz.trimf(self.servico.universe, [0.5, 1.0, 1.0])

        self.gorjeta['pequena'] = fuzz.trimf(self.gorjeta.universe, [0, 0, 10])
        self.gorjeta['media'] = fuzz.trimf(self.gorjeta.universe, [5, 10, 15])
        self.gorjeta['alta'] = fuzz.trimf(self.gorjeta.universe, [10, 20, 20])

    def _definir_regras(self):
        r1 = ctrl.Rule(self.comida['ruim'] | self.servico['ruim'], self.gorjeta['pequena'])
        r2 = ctrl.Rule(self.servico['aceitavel'], self.gorjeta['media'])
        r3 = ctrl.Rule(self.comida['boa'] & self.servico['otima'], self.gorjeta['alta'])
        r4 = ctrl.Rule(self.comida['saborosa'] & self.servico['otima'], self.gorjeta['alta'])
        r5 = ctrl.Rule(self.comida['saborosa'] & self.servico['aceitavel'], self.gorjeta['media'])
        self.regras = [r1, r2, r3, r4, r5]

    def simular(self, comida, servico):
        self.simulador.input['comida'] = comida
        self.simulador.input['servico'] = servico
        self.simulador.compute()
        return self.simulador.output['gorjeta']
    
    def min_gorjeta(self):
        return min(self.gorjeta.universe)
    
    def max_gorjeta(self):
        return max(self.gorjeta.universe)