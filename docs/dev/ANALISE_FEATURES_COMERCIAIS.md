# O que as ferramentas comerciais e os outros CFDs têm que o PINNeAPPle ainda não tem

Análise qualitativa feita em 2026-10-12, só com material público: páginas oficiais, release notes,
blogs e papers. Nenhum produto proprietário passou por engenharia reversa, porque as EULAs proíbem.
O inventário do PINNeAPPle é do commit `68d81d8`. "Tem" quer dizer que o código existe no repo, não
que foi validado no catálogo.

## 1. Os padrões que se repetem no mercado

Seis ideias aparecem em quase todos os produtos de 2025–2026. Elas valem mais do que qualquer feature
isolada.

1. **Surrogate treinado em simulações antigas, sem CAD paramétrico.** É o caso do Ansys SimAI ("non-parametric",
   30–100 resultados), do Simcenter PhysicsAI (lê estudos do Design Manager e STL fora da
   parametrização), de Neural Concept, Luminary SHIFT, Emmi AB-UPT e SimScale Physics AI. Entra
   geometria crua (malha, STL, point cloud) e saem campo 3D + KPIs escalares.
2. **Toda predição vem com um "posso confiar?".** O SimAI tem um *confidence level*; o PhysicsAI e o
   Altair PhysicsAI têm um *geometric similarity score* de 0 a 1 em relação ao treino. Na PhysicsX,
   a UQ dispara active learning.
3. **O surrogate decide qual simulação rodar a seguir.** É o adaptive sampling do HEEDS, o loop de
   UQ → nova simulação da PhysicsX e o active learning da Neural Concept.
4. **ROMs dinâmicos exportados como FMU/ONNX.** Aparece no Twin Builder (Neural ODE, TFT, Hammerstein
   LPV, calibração com sensores e "fusion modeling" do gap físico–medido), no Simcenter ROM Builder
   (LSTM/GRU, FMI + ONNX) e no romAI.
5. **Agentes e MCP por cima de uma API Python.** Existem servidores MCP do PyAnsys, o Mesh Agent e o
   Validation Agent da Ansys, o MCP do SDK do Luminary e o notebook com assistente de código.
6. **Dados com linhagem.** Simulação vira dataset pronto para ML com metadados e lineage; isso aparece
   no PhysicsX Simulation Workbench, no Rescale Data Intelligence, no Teamcenter SPDM e no Granta.

O PINNeAPPle já tem as peças de quase tudo isso, mas várias estão soltas. As maiores lacunas não são de
algoritmo, e sim de **ligação entre módulos**.

## 2. Lacunas, em ordem de prioridade

Critério: quanto o mercado valoriza × quanto encaixa no que já existe ÷ esforço.

### P1: alto valor, encaixa direto

| # | Feature | Quem tem | O que o PINNeAPPle tem hoje | O que falta |
|---|---|---|---|---|
| 1 | **Geometric similarity score por predição** (0–1, "esse design está dentro do treino?") | Simcenter/Altair PhysicsAI, SimAI | `GeometryOODGuardrail`, `trust_gate`, UQ conformal/ensemble | Expor um score único, calibrado, devolvido **junto com cada predição** do surrogate (embedding da geometria → distância ao treino → [0,1]) |
| 2 | **Loop fechado "qual simulação rodar agora"** | HEEDS adaptive sampling, PhysicsX, Neural Concept | `ActiveSampler`, EI do `BayesianDesignOptimizer`, `openfoam/campaign.py`, `pinneapple_registry` | Um orquestrador que liga aquisição → `campaign` → dataset store → re-treino → nova aquisição, com critério de parada por erro de validação |
| 3 | **Surrogate de campo a partir de geometria crua** (STL/point cloud → campo + Cd/Cl) | SimAI, PhysicsAI, Luminary/DoMINO, Emmi AB-UPT | `Transolver`, MeshGraphNet, `physicsnemo_bridge`, `lpm.py` | Um pipeline pronto "pasta de casos OpenFOAM → dataset → treino → relatório", com DoMINO de verdade (o `domino.py` atual é um PINN de decomposição de domínio, e o **nome confunde**) e um benchmark em DrivAerML/AhmedML |
| 4 | **Export de ROM/surrogate como FMU** | Twin Builder, ROM Builder, romAI | ROMs (POD, DMD, OpInf, neural ROM), export ONNX/TorchScript, `fmi_runner` só importa | Empacotar um ROM como FMU co-simulação (pythonfmu ou FMU em C com ONNX Runtime). Fecha o ciclo com o `fmu_inspect` desta rodada |
| 5 | **Servidor MCP do próprio PINNeAPPle** | PyAnsys, Luminary SDK | `PhysicsToolRegistry` + `agent_loop` | Expor o registry como MCP: o Claude, o Cursor e o Kit passam a chamar `solve_pde`, `fit_surrogate` etc. É pouco código, porque o registry já tem schemas |

### P2: alto valor, mais trabalho

| # | Feature | Quem tem | Situação | Observação |
|---|---|---|---|---|
| 6 | **Otimização topológica** (SIMP / level-set) | Mechanical, Discovery, STAR-CCM+ adjoint topology, SU2, OpenFOAM adjoint | **Ausente** | Começar com SIMP 2-D compliance (Sigmund 99 linhas como referência de validação) sobre o `solid_fem.py` |
| 7 | **AMR guiada pela solução** | CONVERGE (a cada passo), Basilisk (wavelet), Luminary SDK (AMR) | **Ausente** (só refinamento estático) | Quadtree com critério de wavelet/gradiente para os solvers 2-D (`fvm`, `shallow_water_fv`) |
| 8 | **Multi-fidelidade** (co-kriging, fusão baixa→alta) | optiSLang/AI+, HEEDS (reuso de dados), COMSOL | **Ausente** (há transfer/meta-learning) | Co-kriging de Kennedy–O'Hagan sobre o GP numpy do `design_optimizer`: LBM grosso + OpenFOAM fino |
| 9 | **ROMs dinâmicos com calibração online** ("fusion modeling") | Twin Builder (Neural ODE, TFT, Hammerstein), ROM Builder (LSTM/GRU) | Neural ROM, Koopman, SINDy, Kalman, 4D-Var | Um modelo "física + resíduo aprendido" calibrado com sensores ao vivo no `digital_twin` |
| 10 | **Polynomial chaos / Sobol para UQ e sensibilidade** | optiSLang, COMSOL UQ | Conformal/ensemble; nenhum PCE | Adicionar PCE com índices de Sobol, barato e com testes de forma fechada (funções de Ishigami) |
| 11 | **Encoders de geometria** (PointNet/DGCNN/UV-Net B-rep) | Neural Concept, PhysicsAI | `geometry_features`, `geometry_intelligence` | Necessário para o item 1 e o 3 funcionarem em geometria arbitrária |

### P3: diferencial de produto, avaliar depois

| # | Feature | Quem tem | Nota |
|---|---|---|---|
| 12 | **Estudo de design como objeto** (Study → Designs → Scenarios, regras que aplicam setup a entidades CAD, "Decision Center" comparando variantes) | Autodesk CFD, COMSOL Model Manager, Simcenter Studio | O `pinneapple_lab` + `registry` está perto; falta a hierarquia e a tela de comparação |
| 13 | **Morphing paramétrico sem CAD** (malha tesselada deformada por vetores, RBF) | STAR-CCM+ 2506, SU2 (RBF, Hicks-Henne) | Habilita otimização em STL de cliente sem acesso ao CAD |
| 14 | **Conector com CAD paramétrico na nuvem** (Onshape Variable Studio → inferência em segundos) | Luminary × Onshape | Hoje: STEP via gmsh e CadQuery. Onshape tem API REST pública |
| 15 | **Viewer remoto com cache de inferência** (re-inferir a cada mudança de UI, cache por hash, diff surrogate × verdade) | Omniverse CFD blueprint, Fluent web UI | O `twin3d` viewer + `pinneapple_app` cobre metade; falta o cache keyed e o diff |
| 16 | **Predição de recurso/tempo de simulação** ("vale rodar em GPU?") | Ansys Electronics AI+, Mechanical | Um regressor treinado nos logs do `campaign` |
| 17 | **Gerativo a partir de KPI** ("quero Cd < X, massa < Y" → geometrias) | PhysicsAI Generate, Neural Concept Copilot, PhysicsX LGM | O `cad_draft` + `genetic_sdf` + surrogate formam o esqueleto |

## 3. Outros softwares de CFD: o que vale emular

| Ideia | De onde | Situação no PINNeAPPle | Vale? |
|---|---|---|---|
| LBM GPU com voxelização automática (sem malhar), D3Q27 cumulant/central-moment, LES com wall model | ultraFluidX, PowerFLOW, XFlow, M-Star, SimScale Pacefish, FluidX3D | `lbm.py` tem D3Q19 BGK + Smagorinsky; `lbm_tunnel` | **Sim**: evoluir para D3Q27 cumulant + voxelização de STL. É o caminho mais curto para um "túnel de vento virtual" nativo |
| Armazenamento FP16 com aritmética FP32 | FluidX3D | — | Sim, metade da memória a custo quase zero (não copiar código, a licença é não comercial) |
| Malha autônoma em tempo de execução + AMR a cada passo | CONVERGE | Ausente | Junto com o item 7 |
| Solver inteiro diferenciável | JAX-Fluids, PhiFlow | `jax_cfd_solver` (bridge), `compute_backends` JAX | Sim, para otimização por gradiente sem adjoint manual |
| Adjoint contínuo/discreto, B-spline volumétrico, MMA/SQP/L-BFGS, multi-ponto | SU2, OpenFOAM ESI `adjointOptimisation` | `ContinuousAdjointSolver` + `ShapeParametrization` | Parcial. Faltam MMA e a parametrização por B-spline volumétrica |
| RANS nativo (k-ω SST, SA) | todos | Só como setup de caso OpenFOAM | Baixa prioridade: o OpenFOAM já faz bem, e um RANS próprio é caro de validar |
| VOF com tensão superficial e AMR wavelet | Basilisk | VOF só via interFoam | Só se houver caso de uso (sloshing, injeção) |
| Alta ordem (flux reconstruction, spectral element) em GPU | PyFR, NekRS | Spectral periódico, `ns3d_spectral` | Não agora |
| Visualização in situ (renderizar enquanto roda) | FluidX3D, NekRS (Ascent), M-Star | Rerun/trame | Rerun já dá isso; falta ligar nos solvers nativos |
| Regressão de resultados entre versões | Amesim (test execution manager) | catálogo de métodos + `simulation_comparator` | O PINNeAPPle **já está à frente** aqui |

## 4. Onde o PINNeAPPle já está à frente

Na pesquisa, nenhum dos produtos documenta publicamente uma governança de V&V como estas:

- **Catálogo de métodos ligado a testes contra referências independentes** (`pinneapple_catalog`,
  `method_status.json`, 89+ itens `validated`).
- **Relatório de evidência e decision record** (`pinneapple_veriphysics`), proveniência assinada
  (`pinneapple_security`) e trust gate com OOD geométrico.
- **Honestidade explícita de escopo** nos módulos ("o que não foi testado e por quê").

Isso é argumento de venda contra SimAI e PhysicsAI, que devolvem um *confidence level* opaco. Vale
transformar em feature visível: o mesmo relatório saindo junto de cada predição de surrogate (liga com
o item 1).

## 5. Próximos passos sugeridos

1. Itens **1 + 3 + 5** como um pacote: "surrogate de geometria crua com score de confiança, via MCP".
   É o que todo comprador compara hoje.
2. Item **2** (loop de simulação ativa) usando o `campaign` do OpenFOAM: transforma o PINNeAPPle de
   "biblioteca" em "fábrica de surrogates".
3. Item **4** (FMU de ROM) para o lado de digital twin e sistemas, que conversa com Twin Builder e
   Amesim.
4. **SIMP** e **PCE/Sobol**: fáceis de validar com forma fechada, entram rápido no catálogo.

## Fontes principais

- Ansys SimAI: https://ansys.synopsys.com/products/ai/simai
- Ansys 2026 R1: https://news.synopsys.com/2026-03-11-Synopsys-Launches-Ansys-2026-R1-to-Re-Engineer-Engineering-with-Joint-Solutions-and-AI-Powered-Products
- Ansys Twin Builder: https://ansys.synopsys.com/products/digital-twin/ansys-twin-builder
- PyAnsys (inclui MCP): https://docs.pyansys.com/
- STAR-CCM+ 2506 / 2510 / 2602: https://blogs.sw.siemens.com/simcenter/simcenter-star-ccm-2602-released/
- Simcenter PhysicsAI: https://news.siemens.com/en-us/siemens-simcenter-physicsai/
- Simcenter ROM Builder: https://www.siemens.com/en-us/products/simcenter/integration-solutions/reduced-order-modeling/
- HEEDS: https://www.siemens.com/en-us/products/simcenter/integration-solutions/heeds/
- Luminary: https://luminary.ai/ , https://luminary.ai/models/ , https://app.luminarycloud.com/docs/api/
- PhysicsX: https://physicsx.ai/platform , https://www.physicsx.ai/newsroom/engineering-in-the-age-of-physics-ai-the-platform-driving-the-shift
- Neural Concept: https://www.neuralconcept.com/press-release/neural-concept-ai-design-copilot
- Godela: https://www.ycombinator.com/launches/NcV-godela-ai-powered-physics-engine-for-engineers
- Omniverse CFD blueprint: https://github.com/NVIDIA-Omniverse-blueprints/digital-twins-for-fluid-simulation
- Emmi AB-UPT: https://arxiv.org/html/2502.09692v4
- SimScale: https://www.simscale.com/product/physics-ai/
- COMSOL 6.3: https://www.comsol.com/release/6.3
- Altair/Siemens ultraFluidX, romAI: https://help.altair.com/vwt/topics/getting_started/overview_ultrafluidx_r.htm , https://www.siemens.com/en-us/products/simcenter/engineering-data-science-ai/romai/
- CONVERGE: https://convergecfd.com/benefits/autonomous-meshing/
- OpenFOAM ESI adjoint: https://www.openfoam.com/documentation/files/adjointOptimisationFoamManual_v2312.pdf
- JAX-Fluids 2.0: https://arxiv.org/html/2406.19494v1 ; PhiFlow: https://proceedings.mlr.press/v235/holl24a.html
- FluidX3D: https://github.com/ProjectPhysX/FluidX3D/blob/master/README.md
