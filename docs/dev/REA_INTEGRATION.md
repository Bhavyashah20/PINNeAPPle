# REA no PINNeAPPle — integração, fluxo de harvest e mapa do ecossistema

Escrito em 2026-10-12. REA = *Reverse Engineer Anything* (github.com/morluto/rea, MIT, v6.4.0 testada).

## 1. Onde o REA encaixa

O REA é a **terceira fonte** do pipeline de pesquisa (`pinneapple_tools.hpo_experiments`):

| Fonte | Traz | Módulo |
|---|---|---|
| arXiv | o método publicado | `hpo_experiments.sources.ArxivSource` |
| GitHub | o repositório | `hpo_experiments.sources.GithubSource` |
| **REA** | **como a feature é construída de fato**, com SHA-256 do artefato e limitações declaradas | `pinneapple_tools.rea_bridge` |

Mesma regra de `pinneapple_llm.research`: tudo rastreia até uma saída real de ferramenta, nunca até a
memória de um LLM.

### O que o REA analisa (e o que não)

| Alvo | REA faz | Neste repo |
|---|---|---|
| JavaScript/Electron, bundles web | grafo semântico estático: funções, call graph, linhas exatas | `rea_bridge.evidence` (`summarize_js_evidence`, `locate`) |
| Binários nativos (.so/.dll/.exe, FMUs compiladas) | pseudocódigo/assembly via Hopper, Ghidra ou IDA | só com um provider instalado (`rea setup`) |
| Python/C++/Julia com fonte | **não analisa semântica**; `import-reference-source` só fixa inventário, linguagens, git HEAD e SHA-256 | proveniência; estudo é leitura direta (`Finding.method="manual"`) |
| Software comercial | tecnicamente possível em binários, mas **as EULAs proíbem engenharia reversa** | não usar; só material público (`reference_only`) |

## 2. Componentes

```
pinneapple_tools/rea_bridge/
  client.py    ReaClient: resolve o executável (PINNEAPPLE_REA_BIN > rea no PATH > npx rea-agents@6.4.0),
               roda comandos com --format json, separa o progresso do stderr
  licenses.py  detect_licenses + policy_for: adapt / clean_room / reference_only
  evidence.py  Evidence JS -> FeatureMap; locate(keywords) -> funções, snippet, callers/callees
  harvest.py   HarvestLedger (JSON) com o gate de licença aplicado na carga
  tools.py     reference_license_policy, rea_analyze_javascript, rea_locate_feature
               no PhysicsToolRegistry (register_all -> _register_research_tools)
.mcp.json      servidor MCP do REA para agentes trabalhando no repo (Claude Code, OpenCode...)
docs/dev/rea_harvest/ledger.json   tudo que foi estudado, sob qual licença, e o que virou
```

### Gate de licença

| Política | Licenças | O que pode entrar |
|---|---|---|
| `adapt` | MIT, BSD, Apache-2.0, ISC, Zlib, CC-BY-4.0, CC0 | código adaptado, com aviso em `NOTICE` |
| `clean_room` | GPL, LGPL, AGPL, MPL, EPL | só o método publicado/comportamento; `adapted` é **recusado** pelo ledger |
| `reference_only` | sem licença, NC/ND, proprietária | nada; só orientação |

Repositório com várias licenças: vale a mais restritiva. LGPL que distribui o texto GPL em `COPYING` é
classificado como LGPL.

## 3. Fluxo de harvest

```python
from pinneapple_tools.rea_bridge import ReaClient, summarize_js_evidence, locate, policy_for_path

pol = policy_for_path("alvo/")                       # 1. licença primeiro
res = ReaClient().analyze_javascript_application("alvo/")
fmap = summarize_js_evidence(res.payload)            # 2. mapa de features
for h in locate(fmap, ["pressure", "vorticity"]):    # 3. onde está a feature
    print(h.function.label, h.callees)
# 4. confirmar se o PINNeAPPle já tem (grep), 5. implementar + teste contra referência independente,
# 6. registrar no ledger (HarvestLedger) com evidence_id / sha256
```

Pelo agente: `PhysicsToolRegistry().register_all()` e `run_agent_loop(...)` já enxergam as tools REA.

## 4. Resultado desta rodada

Implementado (todos com testes contra solução exata, closed form ou identidade):

| Feature | Origem (licença, método) | Arquivo |
|---|---|---|
| ETDRK4 + detector de blow-up | Cox–Matthews/Kassam–Trefethen (clean-room vs. Dedalus GPL); detector do VisualPDE via REA | `numerical_solvers/etdrk4.py` |
| Stable Fluids 2-D | WebGL-Fluid-Simulation via REA (MIT); **projeção corrigida** | `numerical_solvers/stable_fluids.py` |
| Métricas estilo PDEBench | PDEBench (MIT); **fRMSE corrigido** (frequências negativas) | `analysis/validation/field_metrics.py` |
| Curvatura discreta em malhas | PhysicsNeMo (Apache, reimplementação NumPy) | `design/geometry/ops/curvature.py` |

Ideias registradas (não implementadas nesta rodada): iFNO, MultiAdam, modelos PhysicsNeMo ausentes,
ShardTensor, leitor `.rst` do MAPDL, PyFluent, `UsdPhysics` no export USD, simulador no navegador
estilo VisualPDE no `pinneapple_app`.

## 5. Mapa das ferramentas grandes

Licenças conferidas nos arquivos do próprio repositório em 2026-10-12 (commit no ledger).

### NVIDIA / Omniverse

| Ferramenta | Licença | Política | Encaixe no PINNeAPPle |
|---|---|---|---|
| PhysicsNeMo | Apache-2.0 | adapt | backbones (FIGConvNet, TopoDiff, DLWP-HEALPix), mesh calculus (feito), ShardTensor |
| PhysicsNeMo-CFD | Apache-2.0 | adapt | avaliação DrivAerML/Ahmed; métricas Cd/Cl já existem |
| PhysicsNeMo-Sym | Apache-2.0 | adapt | geometria CSG/SDF simbólica — comparar com `pinneapple_physics.symbolic_pde` |
| Warp | Apache-2.0 | adapt | backend diferenciável para `particle_dynamics` (já no ROADMAP) |
| Newton | Apache-2.0 | adapt | já tem bridge |
| PhysX 5 | BSD-3 | adapt | referência de contato/partículas (build nativo) |
| Isaac Sim | Apache-2.0 (repo) | adapt | **agora open source**; o runtime Kit embutido segue os termos NVIDIA |
| Isaac Lab | BSD-3 / Apache | adapt | tarefas sim-to-real para o `pinneapple_worldmodel` |
| OpenUSD | Apache-2.0 (+MIT/BSD) | adapt | adicionar `UsdPhysics` ao export do `pinneapple_twin3d` |
| Omniverse Kit SDK | NVIDIA Software License | reference_only | integrar por arquivo USD + extensão Kit chamando a API REST do `pinneapple_app` |
| Earth-2 Studio | Apache-2.0 | adapt | item *Climate Downscaling* do ROADMAP |
| Cosmos | (já em uso no repo) | — | `physcurator` / world model |

### Solvers e frameworks abertos

| Ferramenta | Licença | Política | Situação |
|---|---|---|---|
| OpenFOAM | GPL-3.0 | clean_room | bridge completo já existe |
| SU2 | LGPL-2.1 | clean_room | adjoint para otimização de forma: só o método |
| FEniCSx (dolfinx) | LGPL-3.0 | clean_room | bridge já existe |
| MOOSE | LGPL-2.1 | clean_room | multifísica acoplada: referência de arquitetura |
| deal.II | Apache-2.0 | adapt | AMR hp; referência para `fem.py` |
| Firedrake | BSD-2 | adapt | geração de código UFL→kernels; ideia para o compilador de resíduos |
| Trixi.jl | MIT | adapt | DG para conservação hiperbólica |
| Oceananigans.jl | MIT | adapt | item *Ocean Physics AI Explorer* |
| Clawpack | BSD-3 | adapt | volumes finitos de alta resolução (`tvd_advection`) |
| Genesis / MuJoCo / Brax / Taichi | Apache-2.0 | adapt | já mapeados ou com bridge |
| ParaView | BSD (arquivo não achado no topo) | reference_only até confirmar | visualização |

### Plataformas comerciais (só material público; EULAs proíbem engenharia reversa)

| Empresa | O que fazem | Equivalente no PINNeAPPle | Caminho aberto |
|---|---|---|---|
| Ansys (Fluent, Mechanical, SimAI, optiSLang) | surrogate em nuvem + DOE/otimização | Surrogate Factory, `design_optimizer` | **PyAnsys (MIT)**: `pymapdl-reader`, `pyfluent` |
| Siemens Simcenter (STAR-CCM+, HEEDS, Amesim) | CFD, busca de design, sistemas 1-D | leitor STAR-CCM+, `pinneapple_systems` | formatos de arquivo |
| Luminary Cloud | SDK Python: Project/Geometry/Mesh/Simulation/Solution, pipelines, surrogates | `pinneapple_app` + `pinneapple_registry` | modelo de recursos como referência de API |
| PhysicsX | Opora (arquiteturas por configuração) + padrões abertos com NVIDIA (GTC 2026) | Arena, model cards | acompanhar o lançamento do tooling aberto |
| Neural Concept | AI Design Copilot (CES 2026): geometria CAD-ready a partir de intenção | `pinneapple_llm.cad_draft`, `pinneapple_design` | — |
| Godela (YC S25) | pergunta em linguagem natural + CAD/dados → modelo físico | `pinneapple_problemdesign`, data assimilation | — |

### Sugestões adicionais para a próxima rodada

Licenças desta lista **ainda não conferidas** — rodar `policy_for_path` antes de qualquer uso.

- **Alvos JS onde o REA rende mais** (análise semântica completa): demos de fluidos/ondas em
  WebGL/WebGPU com licença aberta, e o próprio frontend do `pinneapple_app` (auditoria de
  dependências e do fluxo de dados). Viewers de produtos comerciais (Onshape, Autodesk) **não**:
  termos de uso.
- **FMUs compiladas** (Modelica/FMI): binários nativos dos quais o REA + Ghidra extraem a
  interface e o comportamento de modelos de sistema — encaixa em `external_solvers/modelica`.
  Requer instalar o Ghidra (`rea setup`).
- **Open-source de peso ainda não mapeado**: Nektar++ (spectral/hp), Basilisk (adaptatividade em
  quadtree), JAX-Fluids e PhiFlow (CFD diferenciável), Exponax (ETD em JAX — bom para validar o
  ETDRK4 novo por comparação cruzada), The Well e APEBench (benchmarks).
- **Comerciais com API documentada** que valem leitura (só documentação): COMSOL LiveLink,
  Dassault SIMULIA (scripting `abaqus`, cujo reader já existe), Rescale (API de jobs, útil para o
  `pinneapple_orchestration`).
