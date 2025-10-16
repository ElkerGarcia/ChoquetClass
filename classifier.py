from . import rules_generator  # o el import relativo adecuado
from . import aggregation_functions



class Classifier:
    def __init__(self, variables, reglas_fuzzy, train_data, class_column, q_vector):
        """
        Inicializa el clasificador con un conjunto de reglas y variables lingüísticas.
        """
        self.variables = variables
        self.reglas_fuzzy = reglas_fuzzy
        self.train_data = train_data
        self.class_column = class_column
        self.q_vector = q_vector



    def classify_Choquet(self, patron):
        """
        Clasifica un patrón (diccionario de valores de entrada) 
        devolviendo la clase con mayor grado de solidez.
        """
        # 1. Grado de pertenencia
        GP = rules_generator.GradoPertenencia(self.variables)
        grados_todos = GP.calcular_varias(self.reglas_fuzzy, patron)

        # 2. Asociación fuzzy
        asociacion = rules_generator.Asociacion_fuzzy(self.variables)
        grado_asociacion = asociacion.calcular(
            self.reglas_fuzzy, patron, self.train_data, col_clase=self.class_column
        )

        # 3. Solidez con Choquet
        solidez = rules_generator.SolidezChoquet(self.variables)
        resultado = solidez.solidez_con_choquet(
            patron,
            reglas=self.reglas_fuzzy,
            data=self.train_data,
            col_clase=self.class_column,
            q_vector=self.q_vector
        )

        # 4. Obtener la clase con mayor grado de solidez
        clase_predicha = max(resultado, key=resultado.get)
        return clase_predicha, resultado





    def classify_CC_Choquet(self, patron):

        # 2. Asociación fuzzy
        asociacion = rules_generator.Asociacion_fuzzy(self.variables)
        grado_asociacion = asociacion.calcular(
            self.reglas_fuzzy, patron, self.train_data, col_clase=self.class_column
        )

        # 3. Solidez con Cópula tipo Choquet
        copula=aggregation_functions.CopulaChoquetExponential(self.q_vector)
        resultado=copula.calcular_por_clase(grado_asociacion)


        # 4. Obtener la clase con mayor grado de solidez
        clase_predicha = max(resultado, key=resultado.get)
        return clase_predicha, resultado
        print("Clase predicha:", clase_predicha)
        



    def classify_CF1F2_Choquet(self, patron):

        # 2. Asociación fuzzy
        asociacion = rules_generator.Asociacion_fuzzy(self.variables)
        grado_asociacion = asociacion.calcular(
            self.reglas_fuzzy, patron, self.train_data, col_clase=self.class_column
        )

        # 3. Solidez con Cópula tipo F1F2 Choquet
        copula=aggregation_functions.CF1F2_Integral(self.q_vector)
        resultado=copula.calcular_por_clase(grado_asociacion)


        # 4. Obtener la clase con mayor grado de solidez
        clase_predicha = max(resultado, key=resultado.get)
        return clase_predicha, resultado
        print("Clase predicha:", clase_predicha)
        


    def classify_win_rule(self, patron):
        """
        Determina la clase ganadora según el criterio Winning Rule (WR).

        Parámetro:
        -----------
        grados_asociacion : list[dict]
            Lista de diccionarios con los grados de asociación por clase.

        Retorna:
        --------
        (str, dict)
            - Clase ganadora
            - Diccionario con los puntajes máximos por clase
        """
        asociacion = rules_generator.Asociacion_fuzzy(self.variables)
        grado_asociacion = asociacion.calcular(
            self.reglas_fuzzy, patron, self.train_data, col_clase=self.class_column
        )
        
        wr = aggregation_functions.WinningRuleClassifier()
        puntajes = wr.calcular_puntajes(grado_asociacion)
        clase_ganadora = max(puntajes, key=puntajes.get)
        return clase_ganadora, puntajes
        print("Clase predicha:", clase_ganadora)




    def classify_comb_aditive(self, patron):
        """
        Determina la clase ganadora según el método Additive Combination.

        Retorna:
        --------
        tuple : (clase_ganadora, puntajes_normalizados)
        """
        asociacion = rules_generator.Asociacion_fuzzy(self.variables)
        grado_asociacion = asociacion.calcular(
            self.reglas_fuzzy, patron, self.train_data, col_clase=self.class_column
        ) 
        comb_ad=aggregation_functions.AdditiveCombinationClassifier()
        puntajes = comb_ad.calcular_puntajes(grado_asociacion)
        clase_ganadora = max(puntajes, key=puntajes.get)
        return clase_ganadora, puntajes






