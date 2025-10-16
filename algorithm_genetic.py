from . import rules_generator
from . import aggregation_functions
from . import classifier
from copy import deepcopy
import numpy as np
import pandas as pd
import random



def ajustar_q(q):
    """
    Ajusta cada valor de q según:
    - Si 0.00 < q <= 1.00 → q
    - Si 1.00 < q < 2.00 → 1 / (2 - q)
    """
    q = np.array(q, dtype=float)
    q_ajustado = np.where((q > 1.0) & (q < 2.0), 1 / (2 - q), q)
    return q_ajustado.tolist()



def CR_CC_choquet(test_data, train_data, reglas, variables, col_clase, q_vector):
    """
    Calcula la Tasa de Clasificación (CR) usando las reglas difusas y la integral de Choquet.
    """
    total = len(test_data)
    aciertos = 0
    clf = classifier.Classifier(variables, reglas, train_data, col_clase, q_vector)

    for _, ejemplo in test_data.iterrows():
        clase_predicha, resultado = clf.classify_CC_Choquet(ejemplo)
        clase_real = ejemplo[col_clase]
        if clase_predicha == clase_real:
            aciertos += 1

    CR = aciertos / total
    return CR



def optimizar_q_CC_choquet(test_data, train_data, reglas, variables, col_clase, q_inicial, num_iter=50):
    """
    Optimiza el vector q para maximizar la Tasa de Clasificación (CR)
    """
    clases = list(test_data[col_clase].unique())
    n_clases = len(clases)

    q_best = q_inicial.copy()
    CR_best = CR_CC_choquet(test_data, train_data, reglas, variables, col_clase, ajustar_q(q_best))

    print(f"Iteración 0: CR = {CR_best:.4f}, q = {q_best}")

    for t in range(1, num_iter + 1):
        q_t = np.random.uniform(0.01, 1.99, size=n_clases).tolist()
        CR_t = CR_CC_choquet(test_data, train_data, reglas, variables, col_clase, ajustar_q(q_t))

        if CR_t > CR_best:
            q_best = q_t
            CR_best = CR_t
            print(f"Iteración {t}: Nuevo mejor CR = {CR_best:.4f}, q = {q_best}")

    return ajustar_q(q_best), CR_best



def obtener_mal_clasificados_CC_Choquet(data, variables, train_data, reglas, q_vector_used, col_clase):
    """
    Retorna DataFrame con ejemplos mal clasificados por 'reglas' usando 'q_vector_used'.
    """
    mal = []
    clf = classifier.Classifier(variables, reglas, train_data, col_clase, q_vector_used)

    for _, ejemplo in data.iterrows():
        clase_predicha, resultado = clf.classify_CC_Choquet(ejemplo)
        if len(resultado) == 0:
            continue
        pred = max(resultado, key=resultado.get)
        if pred != ejemplo[col_clase]:
            mal.append(ejemplo)
    return pd.DataFrame(mal)



# ============================================================
# 🔹 ALGORITMO GENÉTICO CON OPTIMIZACIÓN DE q INTEGRADA
# ============================================================

def algoritmo_genetico_CC_Choquet(
    train_data,
    test_data,
    col_clase,
    variables,
    num_rules=20,
    num_iter=10,
    TQ=30,
    cross_prob=0.9
):
    """
    Algoritmo genético para optimizar reglas y q en el enfoque Michigan.
    """
    attributes = [c for c in train_data.columns if c != col_clase]
    classes = list(train_data[col_clase].unique())
    n_clases = len(classes)

    # ---------- Paso 1: Generar población inicial ----------
    rg = rules_generator.RuleGenerator(train_data, attributes, variables, N=num_rules, H=2)
    P0 = rg.generar_reglas(col_clase)
    P0 = [rules_generator.ensure_valid_rule(r, attributes, classes, variables) for r in P0]

    # ---------- Paso 2: Inicializar q ----------
    q_inicial = [1.0] * n_clases
    q_best_used, CR_best = optimizar_q_CC_choquet(test_data, train_data, P0, variables, col_clase, q_inicial, num_iter=TQ)
    P_best = deepcopy(P0)

    historial = [(0, CR_best, q_best_used.copy())]
    print(f"Optimización inicial completada: CR = {CR_best:.4f}, q_used = {q_best_used}")

    # ---------- Paso 3: Evolución genética ----------
    fitness_model = rules_generator.FitnessSimple(variables)

    for iteration in range(1, num_iter + 1):
        print(f"\n--- Iteración {iteration} ---")

        # Copiar la mejor población
        P_iter = deepcopy(P_best)

        # Calcular fitness
        fitness_dict = fitness_model.calcular_fitness_todas(P_iter, train_data, col_clase)

        # Seleccionar las peores reglas
        Nreplace = max(1, len(P_iter) // 2)
        peor_indices = sorted(fitness_dict, key=fitness_dict.get)[:Nreplace]

        # Generar nuevas reglas
        nuevas_reglas = []

        # (1) Cruzamiento y mutación
        n_genetic = Nreplace // 2
        for _ in range(n_genetic):
            parent1, parent2 = random.sample(P_iter, 2)
            child = rules_generator.crossover_reglas(parent1, parent2, cross_prob, attributes, classes, variables)
            child = rules_generator.mutacion_regla(child, variables, mutation_prob=1/len(attributes), attributes=attributes, classes=classes)
            nuevas_reglas.append(child)

        # (2) Reglas a partir de ejemplos mal clasificados
        n_mpb = Nreplace - n_genetic
        misclassified = obtener_mal_clasificados_CC_Choquet(train_data, variables, train_data, P_iter, q_best_used, col_clase)
        for _ in range(n_mpb):
            if not misclassified.empty:
                ejemplo_base = misclassified.sample(1).iloc[0]
                new_rule = rules_generator.regla_desde_ejemplo(ejemplo_base, variables, attributes, col_clase)
            else:
                new_rule, _ = rg.generar_regla(col_clase)
            new_rule = rules_generator.ensure_valid_rule(new_rule, attributes, classes, variables)
            nuevas_reglas.append(new_rule)

        # Reemplazar peores reglas
        for idx, new_rule in zip(peor_indices, nuevas_reglas):
            P_iter[idx] = new_rule

        # ---- Optimizar q para la nueva población ----
        q_best_iter, CR_iter = optimizar_q_CC_choquet(
            test_data, train_data, P_iter, variables, col_clase, q_best_used, num_iter=TQ
        )

        print(f"Iter {iteration}: CR optimizado = {CR_iter:.4f}, q_used = {q_best_iter}")

        # Si mejora, actualizar global
        if CR_iter > CR_best:
            CR_best = CR_iter
            q_best_used = q_best_iter
            P_best = deepcopy(P_iter)
            print(f"✅ Iter {iteration}: Nueva mejor población, CR = {CR_best:.4f}")

        historial.append((iteration, CR_best, q_best_used.copy()))

    print(f"\n🏁 Mejor CR final: {CR_best:.4f}")
    print(f"Mejor q_used encontrado: {q_best_used}")
    return P_best, q_best_used, historial








 


# ============================================================
# 🔹 CR con CF1F2-Choquet
# ============================================================
def CR_CF1F2_choquet(test_data, train_data, reglas, variables, col_clase, q_vector):
    """
    Calcula la Tasa de Clasificación (CR) usando las reglas difusas
    y la integral CF1F2-Choquet.
    """
    total = len(test_data)
    aciertos = 0
    clf = classifier.Classifier(variables, reglas, train_data, col_clase, q_vector)

    for _, ejemplo in test_data.iterrows():
        clase_predicha, resultado = clf.classify_CF1F2_Choquet(ejemplo)
        if clase_predicha == ejemplo[col_clase]:
            aciertos += 1

    return aciertos / total


# ============================================================
# 🔹 Ajuste de q
# ============================================================
def ajustar_q(q):
    """
    Ajusta cada valor de q según:
    - Si 0.00 < q <= 1.00 → q
    - Si 1.00 < q < 2.00 → 1 / (2 - q)
    """
    q = np.array(q, dtype=float)
    q_ajustado = np.where((q > 1.0) & (q < 2.0), 1 / (2 - q), q)
    return q_ajustado.tolist()


# ============================================================
# 🔹 Optimización de q (solo cruza consecuentes)
# ============================================================
def optimizar_q_CF1F2_choquet(test_data, train_data, reglas, variables, col_clase, q_inicial, num_iter=50, cross_prob=0.9):
    """
    Optimiza el vector q para maximizar la Tasa de Clasificación (CR)
    en el modelo CF1F2-Choquet.
    Solo cruza los valores de q asociados a las clases (consecuentes).
    """
    clases = list(test_data[col_clase].unique())
    n_clases = len(clases)

    q_best = ajustar_q(q_inicial)
    CR_best = CR_CF1F2_choquet(test_data, train_data, reglas, variables, col_clase, q_best)
    print(f"Iteración 0: CR = {CR_best:.4f}, q = {q_best}")

    for t in range(1, num_iter + 1):
        q_t = q_best.copy()

        # 🔹 Cruce entre valores de q (solo los consecuentes)
        if random.random() < cross_prob:
            i, j = random.sample(range(n_clases), 2)
            q_t[i], q_t[j] = q_t[j], q_t[i]

        # 🔹 Mutación aleatoria suave
        if random.random() < 0.3:
            idx = random.randrange(n_clases)
            q_t[idx] = np.clip(q_t[idx] + np.random.uniform(-0.2, 0.2), 0.01, 1.99)

        q_t = ajustar_q(q_t)
        CR_t = CR_CF1F2_choquet(test_data, train_data, reglas, variables, col_clase, q_t)

        if CR_t > CR_best:
            q_best, CR_best = q_t, CR_t
            print(f"Iter {t}: Nuevo mejor CR = {CR_best:.4f}, q = {q_best}")

    return ajustar_q(q_best), CR_best


# ============================================================
# 🔹 Ejemplos mal clasificados
# ============================================================
def obtener_mal_clasificados_CF1F2_Choquet(data, variables, train_data, reglas, q_vector_used, col_clase):
    """
    Retorna DataFrame con ejemplos mal clasificados por 'reglas'
    usando 'q_vector_used'.
    """
    mal = []
    clf = classifier.Classifier(variables, reglas, train_data, col_clase, q_vector_used)

    for _, ejemplo in data.iterrows():
        clase_predicha, resultado = clf.classify_CF1F2_Choquet(ejemplo)
        if len(resultado) == 0:
            continue
        if clase_predicha != ejemplo[col_clase]:
            mal.append(ejemplo)

    return pd.DataFrame(mal)


# ============================================================
# 🔹 Algoritmo Genético CF1F2-Choquet
# ============================================================
def algoritmo_genetico_CF1F2_Choquet(
    train_data,
    test_data,
    col_clase,
    variables,
    num_rules=20,
    num_iter=10,
    TQ=30,
    cross_prob=0.9
):
    """
    Algoritmo genético para optimizar reglas y q en el enfoque Michigan
    con la integral CF1F2-Choquet.
    """
    attributes = [c for c in train_data.columns if c != col_clase]
    classes = list(train_data[col_clase].unique())
    n_clases = len(classes)

    # ---------- Paso 1: Generar población inicial ----------
    rg = rules_generator.RuleGenerator(train_data, attributes, variables, N=num_rules, H=2)
    P0 = rg.generar_reglas(col_clase)
    P0 = [rules_generator.ensure_valid_rule(r, attributes, classes, variables) for r in P0]

    # ---------- Paso 2: Inicializar q ----------
    q_inicial = [1.0] * n_clases
    q_best_used, CR_best = optimizar_q_CF1F2_choquet(
        test_data, train_data, P0, variables, col_clase, q_inicial, num_iter=TQ
    )
    P_best = deepcopy(P0)

    historial = [(0, CR_best, q_best_used.copy())]
    print(f"Optimización inicial completada: CR = {CR_best:.4f}, q_used = {q_best_used}")

    # ---------- Paso 3: Evolución genética ----------
    fitness_model = rules_generator.FitnessSimple(variables)

    for iteration in range(1, num_iter + 1):
        print(f"\n--- Iteración {iteration} ---")

        P_iter = deepcopy(P_best)
        fitness_dict = fitness_model.calcular_fitness_todas(P_iter, train_data, col_clase)

        # Reemplazar las peores reglas
        Nreplace = max(1, len(P_iter) // 2)
        peor_indices = sorted(fitness_dict, key=fitness_dict.get)[:Nreplace]

        nuevas_reglas = []

        # (1) Cruzamiento y mutación
        n_genetic = Nreplace // 2
        for _ in range(n_genetic):
            parent1, parent2 = random.sample(P_iter, 2)
            child = rules_generator.crossover_reglas(parent1, parent2, cross_prob, attributes, classes, variables)
            child = rules_generator.mutacion_regla(child, variables, mutation_prob=1/len(attributes),
                                                   attributes=attributes, classes=classes)
            nuevas_reglas.append(child)

        # (2) A partir de ejemplos mal clasificados
        n_mpb = Nreplace - n_genetic
        misclassified = obtener_mal_clasificados_CF1F2_Choquet(train_data, variables, train_data, P_iter, q_best_used, col_clase)
        for _ in range(n_mpb):
            if not misclassified.empty:
                ejemplo_base = misclassified.sample(1).iloc[0]
                new_rule = rules_generator.regla_desde_ejemplo(ejemplo_base, variables, attributes, col_clase)
            else:
                new_rule, _ = rg.generar_regla(col_clase)
            new_rule = rules_generator.ensure_valid_rule(new_rule, attributes, classes, variables)
            nuevas_reglas.append(new_rule)

        # Reemplazar reglas
        for idx, new_rule in zip(peor_indices, nuevas_reglas):
            P_iter[idx] = new_rule

        # ---- Optimizar q para la nueva población ----
        q_best_iter, CR_iter = optimizar_q_CF1F2_choquet(
            test_data, train_data, P_iter, variables, col_clase, q_best_used, num_iter=TQ
        )

        print(f"Iter {iteration}: CR optimizado = {CR_iter:.4f}, q_used = {q_best_iter}")

        # Si mejora, actualizar global
        if CR_iter > CR_best:
            CR_best = CR_iter
            q_best_used = q_best_iter
            P_best = deepcopy(P_iter)
            print(f"✅ Iter {iteration}: Nueva mejor población, CR = {CR_best:.4f}")

        historial.append((iteration, CR_best, q_best_used.copy()))

    print(f"\n🏁 Mejor CR final: {CR_best:.4f}")
    print(f"Mejor q_used encontrado: {q_best_used}")
    return P_best, q_best_used, historial







# ============================================================
# 🔹 CR con win-rule
# ============================================================
def CR_win_rule(test_data, train_data, reglas, variables, col_clase, q_vector):
    """
    Calcula la Tasa de Clasificación (CR) usando las reglas difusas y el método Win Rule.
    """
    total = len(test_data)
    aciertos = 0
    clf = classifier.Classifier(variables, reglas, train_data, col_clase, q_vector)

    for _, ejemplo in test_data.iterrows():
        clase_predicha, resultado = clf.classify_win_rule(ejemplo)
        if clase_predicha == ejemplo[col_clase]:
            aciertos += 1

    return aciertos / total


# ============================================================
# 🔹 Ajuste de q
# ============================================================
def ajustar_q(q):
    """
    Ajusta cada valor de q según:
    - Si 0.00 < q <= 1.00 → q
    - Si 1.00 < q < 2.00 → 1 / (2 - q)
    """
    q = np.array(q, dtype=float)
    q_ajustado = np.where((q > 1.0) & (q < 2.0), 1 / (2 - q), q)
    return q_ajustado.tolist()


# ============================================================
# 🔹 Optimización de q (solo cruza consecuentes)
# ============================================================
def optimizar_q_win_rule(test_data, train_data, reglas, variables, col_clase, q_inicial, num_iter=50, cross_prob=0.9):
    """
    Optimiza el vector q para maximizar la Tasa de Clasificación (CR)
    en el modelo Win Rule.
    Solo cruza los valores de q asociados a las clases (consecuentes).
    """
    clases = list(test_data[col_clase].unique())
    n_clases = len(clases)

    q_best = ajustar_q(q_inicial)
    CR_best = CR_win_rule(test_data, train_data, reglas, variables, col_clase, q_best)
    print(f"Iteración 0: CR = {CR_best:.4f}, q = {q_best}")

    for t in range(1, num_iter + 1):
        q_t = q_best.copy()

        # 🔹 Cruce entre valores de q (solo los consecuentes)
        if random.random() < cross_prob:
            i, j = random.sample(range(n_clases), 2)
            q_t[i], q_t[j] = q_t[j], q_t[i]

        # 🔹 Mutación aleatoria suave
        if random.random() < 0.3:
            idx = random.randrange(n_clases)
            q_t[idx] = np.clip(q_t[idx] + np.random.uniform(-0.2, 0.2), 0.01, 1.99)

        q_t = ajustar_q(q_t)
        CR_t = CR_win_rule(test_data, train_data, reglas, variables, col_clase, q_t)

        if CR_t > CR_best:
            q_best, CR_best = q_t, CR_t
            print(f"Iter {t}: Nuevo mejor CR = {CR_best:.4f}, q = {q_best}")

    return ajustar_q(q_best), CR_best


# ============================================================
# 🔹 Ejemplos mal clasificados (para MPB)
# ============================================================
def obtener_mal_clasificados_win_rule(data, variables, train_data, reglas, q_vector_used, col_clase):
    """
    Retorna DataFrame con ejemplos mal clasificados por 'reglas' usando 'q_vector_used'.
    """
    mal = []
    clf = classifier.Classifier(variables, reglas, train_data, col_clase, q_vector_used)

    for _, ejemplo in data.iterrows():
        clase_predicha, resultado = clf.classify_win_rule(ejemplo)
        if len(resultado) == 0:
            continue
        if clase_predicha != ejemplo[col_clase]:
            mal.append(ejemplo)

    return pd.DataFrame(mal)


# ============================================================
# 🔹 Algoritmo Genético Win Rule
# ============================================================
def algoritmo_genetico_win_rule(
    train_data,
    test_data,
    col_clase,
    variables,
    num_rules=20,
    num_iter=10,
    TQ=30,
    cross_prob=0.9
):
    """
    Algoritmo genético para optimizar reglas y q en el enfoque Michigan
    con el método Win Rule.
    """
    attributes = [c for c in train_data.columns if c != col_clase]
    classes = list(train_data[col_clase].unique())
    n_clases = len(classes)

    # ---------- Paso 1: Generar población inicial ----------
    rg = rules_generator.RuleGenerator(train_data, attributes, variables, N=num_rules, H=2)
    P0 = rg.generar_reglas(col_clase)
    P0 = [rules_generator.ensure_valid_rule(r, attributes, classes, variables) for r in P0]

    # ---------- Paso 2: Inicializar q ----------
    q_inicial = [1.0] * n_clases
    q_best_used, CR_best = optimizar_q_win_rule(
        test_data, train_data, P0, variables, col_clase, q_inicial, num_iter=TQ
    )
    P_best = deepcopy(P0)
    historial = [(0, CR_best, q_best_used.copy())]
    print(f"Optimización inicial completada: CR = {CR_best:.4f}, q_used = {q_best_used}")

    # ---------- Paso 3: Evolución genética ----------
    fitness_model = rules_generator.FitnessSimple(variables)

    for iteration in range(1, num_iter + 1):
        print(f"\n--- Iteración {iteration} ---")

        P_iter = deepcopy(P_best)
        fitness_dict = fitness_model.calcular_fitness_todas(P_iter, train_data, col_clase)

        Nreplace = max(1, len(P_iter) // 2)
        peor_indices = sorted(fitness_dict, key=fitness_dict.get)[:Nreplace]

        nuevas_reglas = []

        # (1) Reglas por operadores genéticos
        n_genetic = Nreplace // 2
        for _ in range(n_genetic):
            parent1, parent2 = random.sample(P_iter, 2)
            child = rules_generator.crossover_reglas(parent1, parent2, cross_prob, attributes, classes, variables)
            child = rules_generator.mutacion_regla(child, variables, mutation_prob=1/len(attributes),
                                                   attributes=attributes, classes=classes)
            nuevas_reglas.append(child)

        # (2) Reglas basadas en ejemplos mal clasificados
        n_mpb = Nreplace - n_genetic
        misclassified = obtener_mal_clasificados_win_rule(train_data, variables, train_data, P_iter, q_best_used, col_clase)
        for _ in range(n_mpb):
            if not misclassified.empty:
                ejemplo_base = misclassified.sample(1).iloc[0]
                new_rule = rules_generator.regla_desde_ejemplo(ejemplo_base, variables, attributes, col_clase)
            else:
                new_rule, _ = rg.generar_regla(col_clase)
            new_rule = rules_generator.ensure_valid_rule(new_rule, attributes, classes, variables)
            nuevas_reglas.append(new_rule)

        # Reemplazar las peores reglas
        for idx, new_rule in zip(peor_indices, nuevas_reglas):
            P_iter[idx] = new_rule

        # ---- Optimizar q para esta nueva población ----
        q_best_iter, CR_iter = optimizar_q_win_rule(
            test_data, train_data, P_iter, variables, col_clase, q_best_used, num_iter=TQ
        )

        print(f"Iter {iteration}: CR optimizado = {CR_iter:.4f}, q_used = {q_best_iter}")

        # Si mejora, actualizar global
        if CR_iter > CR_best:
            CR_best = CR_iter
            q_best_used = q_best_iter
            P_best = deepcopy(P_iter)
            print(f"✅ Iter {iteration}: Nueva mejor población, CR = {CR_best:.4f}")

        historial.append((iteration, CR_best, q_best_used.copy()))

    print(f"\n🏁 Mejor CR final: {CR_best:.4f}")
    print(f"Mejor q_used encontrado: {q_best_used}")
    return P_best, q_best_used, historial






# ================================================================
#      ALGORITMO GENÉTICO - COMBINACIÓN ADITIVA
# ================================================================

def CR_comb_aditive(test_data, train_data, reglas, variables, col_clase, q_vector):
    """
    Calcula la Tasa de Clasificación (CR) usando las reglas difusas
    con combinación aditiva.
    """
    total = len(test_data)
    aciertos = 0
    clf = classifier.Classifier(variables, reglas, train_data, col_clase, q_vector)

    for _, ejemplo in test_data.iterrows():
        clase_predicha, resultado = clf.classify_comb_aditive(ejemplo)
        clase_real = ejemplo[col_clase]
        if clase_predicha == clase_real:
            aciertos += 1

    return aciertos / total


def optimizar_q_comb_aditive(test_data, train_data, reglas, variables, col_clase, q_inicial, num_iter=50):
    """
    Optimiza el vector q para maximizar la tasa de clasificación (CR)
    usando la combinación aditiva.
    """
    clases = list(test_data[col_clase].unique())
    n_clases = len(clases)

    q_best = q_inicial.copy()
    CR_best = CR_comb_aditive(
        test_data=test_data,
        train_data=train_data,
        reglas=reglas,
        variables=variables,
        col_clase=col_clase,
        q_vector=ajustar_q(q_best)
    )

    print(f"Iteración 0: CR = {CR_best:.4f}, q = {q_best}")

    for t in range(1, num_iter + 1):
        q_t = np.random.uniform(0.01, 1.99, size=n_clases).tolist()
        CR_t = CR_comb_aditive(
            test_data=test_data,
            train_data=train_data,
            reglas=reglas,
            variables=variables,
            col_clase=col_clase,
            q_vector=ajustar_q(q_t)
        )

        if CR_t > CR_best:
            q_best = q_t
            CR_best = CR_t
            print(f"Iteración {t}: Nuevo mejor CR = {CR_best:.4f}, q = {q_best}")

    return ajustar_q(q_best), CR_best


def obtener_mal_clasificados_comb_aditive(data, variables, train_data, reglas, q_vector_used, col_clase):
    """
    Retorna DataFrame con ejemplos mal clasificados usando combinación aditiva.
    """
    mal = []
    clf = classifier.Classifier(variables, reglas, train_data, col_clase, q_vector_used)

    for _, ejemplo in data.iterrows():
        clase_predicha, resultado = clf.classify_comb_aditive(ejemplo)
        if len(resultado) == 0:
            continue
        pred = max(resultado, key=resultado.get)
        if pred != ejemplo[col_clase]:
            mal.append(ejemplo)

    return pd.DataFrame(mal)


# ================================================================
#      ALGORITMO GENÉTICO PRINCIPAL - COMBINACIÓN ADITIVA
# ================================================================

def algoritmo_genetico_comb_aditive(
    train_data,
    test_data,
    col_clase,
    variables,
    num_rules=20,
    num_iter=10,
    TQ=30,
    cross_prob=0.9
):
    """
    Algoritmo genético basado en combinación aditiva
    (enfoque Michigan).
    """
    attributes = [c for c in train_data.columns if c != col_clase]
    classes = list(train_data[col_clase].unique())
    n_clases = len(classes)

    # ----- Generar población inicial -----
    rg = rules_generator.RuleGenerator(train_data, attributes, variables, N=num_rules, H=2)
    P0 = rg.generar_reglas(col_clase)
    P0 = [rules_generator.ensure_valid_rule(r, attributes, classes, variables) for r in P0]

    # ----- Inicializar q y calcular CR inicial -----
    q_best = [1.0] * n_clases
    q_best_used = ajustar_q(q_best)
    CR_best = CR_comb_aditive(test_data, train_data, P0, variables, col_clase, q_best_used)
    P_best = deepcopy(P0)

    historial = [(0, CR_best, q_best_used.copy())]
    print(f"Iteración inicial: CR = {CR_best:.4f}, q_used = {q_best_used}")

    # ----- Optimización inicial de q -----
    for t in range(1, TQ + 1):
        q_random = np.random.uniform(0.01, 1.99, size=n_clases).tolist()
        q_random_used = ajustar_q(q_random)
        CR_t = CR_comb_aditive(test_data, train_data, P0, variables, col_clase, q_random_used)
        if CR_t > CR_best:
            q_best = q_random
            q_best_used = q_random_used
            CR_best = CR_t
            P_best = deepcopy(P0)
            print(f"TQ {t}: Nuevo mejor CR = {CR_best:.4f}, q_used = {q_best_used}")

    historial.append(("TQ", CR_best, q_best_used.copy()))

    # ----- Evolución genética -----
    fitness_model = rules_generator.FitnessSimple(variables)

    for iteration in range(1, num_iter + 1):
        print(f"\n--- Iteración {iteration} ---")

        P_iter = deepcopy(P_best)
        fitness_dict = fitness_model.calcular_fitness_todas(P_iter, train_data, col_clase)

        Nreplace = max(1, len(P_iter) // 2)
        peor_indices = sorted(fitness_dict, key=fitness_dict.get)[:Nreplace]

        nuevas_reglas = []

        # --- 1) Operadores genéticos ---
        n_genetic = Nreplace // 2
        for _ in range(n_genetic):
            parent1, parent2 = random.sample(P_iter, 2)
            child = rules_generator.crossover_reglas(parent1, parent2, cross_prob, attributes, classes, variables)
            child = rules_generator.mutacion_regla(child, variables, mutation_prob=1/len(attributes), attributes=attributes, classes=classes)
            nuevas_reglas.append(child)

        # --- 2) MPB (ejemplos mal clasificados) ---
        n_mpb = Nreplace - n_genetic
        misclassified = obtener_mal_clasificados_comb_aditive(train_data, variables, train_data, P_iter, q_best_used, col_clase)
        for _ in range(n_mpb):
            if not misclassified.empty:
                ejemplo_base = misclassified.sample(1).iloc[0]
                new_rule = rules_generator.regla_desde_ejemplo(ejemplo_base, variables, attributes, col_clase)
            else:
                new_rule, _ = rg.generar_regla(col_clase)
            new_rule = rules_generator.ensure_valid_rule(new_rule, attributes, classes, variables)
            nuevas_reglas.append(new_rule)

        # --- Reemplazar peores reglas ---
        for idx, new_rule in zip(peor_indices, nuevas_reglas):
            P_iter[idx] = new_rule

        # --- Evaluar población actual ---
        CR_iter = CR_comb_aditive(test_data, train_data, P_iter, variables, col_clase, q_best_used)
        print(f"CR con q_best_used: {CR_iter:.4f}")

        if CR_iter > CR_best:
            CR_best = CR_iter
            P_best = deepcopy(P_iter)
            print(f"Iter {iteration}: ¡Nueva mejor población! CR = {CR_best:.4f}")

        # --- Optimizar q para esta población ---
        for t in range(TQ):
            q_random = np.random.uniform(0.01, 1.99, size=n_clases).tolist()
            q_random_used = ajustar_q(q_random)
            CR_t = CR_comb_aditive(test_data, train_data, P_iter, variables, col_clase, q_random_used)
            if CR_t > CR_best:
                q_best = q_random
                q_best_used = q_random_used
                CR_best = CR_t
                P_best = deepcopy(P_iter)
                print(f"Iter {iteration} - TQ {t}: Mejor CR = {CR_best:.4f}, q_used = {q_best_used}")

        historial.append((iteration, CR_best, q_best_used.copy()))

    print(f"\nMejor CR final: {CR_best:.4f}")
    print(f"Mejor q_used encontrado: {q_best_used}")
    return P_best, q_best_used, historial






