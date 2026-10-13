---
title: "Leitor de resultados .rst do MAPDL e driver PyFluent"
labels: [enhancement, interop]
---

**Quem tem:** PyAnsys `pymapdl-reader` e `pyfluent` (MIT).

**O que o PINNeAPPle já tem:** Readers de Abaqus, Exodus, CGNS, Fluent mesh, STAR-CCM+; Fluent via journal.

**Proposta:** Reader `.rst` (via pymapdl-reader opcional) para UPD; caminho PyFluent no `external_solvers/ansys`.

**Critério de aceite (regra D1 — referência independente):** Arquivo `.rst` de exemplo do próprio pymapdl-reader lido com deslocamentos iguais aos da biblioteca de referência.

**Origem:** ledger: PyAnsys
