# ESA SpOC-3 — Faza 7: rezultatet e verifikuara

16 nëntor 2026 · Projekt universitar, Dizajni dhe Analiza e Algoritmeve, FIEK

## Rezultati

Faza 7 është implementuar dhe testuar mbi branch-in `feat/random-search`, commit
`355adee126cff1ab9c4d2ece368e25042b542b7f`, të verifikuar drejtpërdrejt në GitHub.
Puna është në branch-in `work/advanced-optimization`, pa push ose Pull Request.
Të **32 skedarët ekzistues** të bazës janë identikë byte për byte; përmirësimet
janë shtesa. Dy ZIP-at u përdorën për referencë/histori dhe rezultate historike,
jo për bashkim mekanik të kodit.

**Të tri objektivat ambicioze janë tejkaluar.** Vlerat më negative janë më të mira.

| Instanca | Më e mira historike, 2,000 vlerësime/run | Më e mira e re, e verifikuar | Synimi | Buxheti i run-it fitues |
|---|---:|---:|---:|---:|
| Beginner | -0.020408163265 | **-0.142857142857** | -0.10 | 1,500 |
| Intermediate | -0.156454491374 | **-1.723378941106** | -0.80 | 50,000 |
| Hard | -0.023200475907 | **-1.372397382510** | -0.50 | 50,000 |

Tabela e mësipërme dallon rekordin historik nga rekordi i ri; buxhetet ndryshojnë.
Krahasimi i drejtë me buxhete të barabarta jepet veçmas më poshtë. Seed-i,
algoritmi, momenti i gjetjes dhe hash-i i secilit kromozom ruhen te
`evidence/phase7/best-index.json`. Kromozomet janë te `evidence/phase7/best/`.
Secili është rilexuar nga JSON dhe kontrolluar sërish me wrapper-in dhe me
`udp.fitness()` të ESA-s: përputhje e saktë e vlerës së ruajtur.

Për Hard, rekordi me 10,000 vlerësime dhe ai me 50,000 kanë të njëjtin mbulim
diskret. Vlerat zyrtare ndryshojnë vetëm rreth 2.22×10⁻¹⁶ nga rrumbullakimi i
mbledhjes së tri projeksioneve. Eksportohet kandidati me vlerën numerikisht më
të ulët, por ky ndryshim nuk trajtohet si përmirësim shkencor i cilësisë.

## Pse metodat e mëparshme ngecnin

Evaluatori zyrtar minimizon:

\[
f(x)=-\min_{k\in\{0,1,2\}}\left(F_{XY,k}+F_{XZ,k}+F_{YZ,k}\right).
\]

Çdo satelit ka gjashtë komponente fillestare. STM-të e ESA-s i propagojnë ato
në tri kohë vëzhgimi. Koordinatat fillestare kanë kufij ±1 dhe shpejtësitë ±10.
Shpejtësi uniforme në këtë interval shpesh japin pozicione shumë të mëdha në
vëzhgimet e mëvonshme. Sateliti hiqet nëse qoftë edhe një koordinatë është
jashtë intervalit të hapur (−1,1), përpara projeksionit në XY/XZ/YZ. Nëse një
vëzhgim mbetet pa satelitë, shuma e tij është zero dhe fitness-i i gjithë
formacionit bëhet zero, edhe kur vëzhgimet e tjera janë më të mira.

Pas propagimit, vëzhgimet e dyta dhe të treta pjesëtohen secila vetëm një herë
me 1.23. Rrjeta është 11×11 për Beginner dhe 21×21 për dy instancat e tjera.
Konvertimi në qelizë bëhet pas zhvendosjes së koordinatës; nuk ndryshuam as
rendin e këtyre veprimeve, as autocorrelation-in, as formulën e fitness-it.

Brenda së njëjtës qelizë, perturbimet e vogla mund të mos ndryshojnë asgjë:
objektivi ka pllaja. DE me diferenca të mëdha në hapësirën origjinale dhe
crossover sipas gjeneve shpesh humbet lidhjen pozicion–shpejtësi. GA-ja e vjetër
ka një avantazh të mundshëm nga kombinimet që tkurrin shpejtësitë drejt qendrës,
por ende kërkon kryesisht në rajone me pak satelitë të mbijetuar. Kjo është
shpjegim mekanistik i mbështetur nga kodi dhe matjet, jo provë që çdo operator
i GA-së është individualisht shkaku i përmirësimit.

## Inicializimi i informuar: SVD mbi kufizimet STM

Për një satelit me gjendje \(s\in\mathbb{R}^6\), shënojmë me \(A_k\) tri
rreshtat e pozicionit të STM-së, të pjesëtuar me inflacionin kur \(k>0\).
Shkalla fizike `1e-4` anulohet matematikisht në formulën lineare; evaluatori
zyrtar vazhdon t'i kryejë veprimet e tij origjinale numerike.

Ndërtojmë matricën 15×6:

\[
C=\begin{bmatrix}
\operatorname{diag}(1,1,1,0.1,0.1,0.1)\\ A_0\\ A_1\\ A_2
\end{bmatrix},\qquad C=U\Sigma V^T,\qquad s=Wz,\quad W=V\Sigma^{-1}.
\]

Kushti \(\lVert Cs\rVert_\infty\le0.995\) siguron kufijtë e vendimit dhe
mbijetesën në **tri momentet zyrtare**. Nuk pretendohet mbijetesë në çdo çast
ndërmjet tyre. Me këtë transformim, kolonat e \(CW\) janë ortonormale dhe
kërkimi nuk dominohet nga drejtimet me amplifikim të madh orbital.

Mostrojmë një drejtim Gaussian gjashtëdimensional, e shkallëzojmë deri te faqja
e rajonit të lejueshëm dhe përdorim rreze me faktor \(r^{1/6}\), ku \(r\sim\mathcal U(0,1)\). Çdo satelit
mostrohet veçmas. Kjo është shpërndarje e informuar dhe e shpërndarë në rajon;
**nuk është pretendim për shpërndarje uniforme në të gjithë politopin**.

Nuk përdoren inverse të ekuacioneve normale ose pseudoinverse të prera.
Numri i kushtëzimit është rreth 166.70 për DRO dhe 127.62 për HALO. Kodi kontrollon
vlerat singulare, mbetjen e ortogonalitetit, NaN/Inf dhe sisteme tepër të
keqkushtëzuara. Mbrojtja 0.995 jep hapësirë për gabimet e rrumbullakimit.

Studimi i veçantë përdori 300 mostra për metodë/instancë, seed-et 901/902/903:

| Instanca | Fitness negativ: uniform | Fitness negativ: STM | Satelitë mesatarë me STM në t0/t1/t2 |
|---|---:|---:|---|
| Beginner | 0.00% | 100.00% | 5.0/5.0/5.0 |
| Intermediate | 1.33% | 100.00% | 40.0/40.0/40.0 |
| Hard | 0.00% | 100.00% | 40.0/40.0/40.0 |

Këto janë frekuenca empirike të këtij studimi. Të dhënat individuale ruhen në
`initialization/samples.csv`; u shpenzuan 1,800 vlerësime zyrtare. Kërkimi ynë
zgjedh një nënrajon të lejueshëm të hapësirës së kërkimit; kufijtë zyrtarë nuk ndryshohen.
Ky kufizim mund të përjashtojë zgjidhje më të mira që humbin disa satelitë.

## Algoritmet dhe çfarë treguan provat

- **STM random** izolon efektin e inicializimit. Për Beginner, ai mjafton për
  të arritur kufirin optimal. Për dy instancat e mëdha, përmirësimi shtesë vjen
  nga optimizimi i mbulimit të projeksioneve.
- **STM GA** përdor popullatë 40, tournament 2, dy elita dhe crossover 0.9.
  Crossover-i shkëmben gjashtë komponentet e një sateliti së bashku. Mutacioni
  ndjek një drejtim Gaussian në koordinatat e transformuara dhe një hap të
  lejueshëm të matur në qeliza; herë pas here zëvendëson satelitin. Shkalla
  përshtatet gjatë stagnimit; pas 20 brezash pa përmirësim rifreskohet gjysma
  më e dobët. Elitat ripërdorin fitness-in e ruajtur.
- **STM DE** përdor `current-to-pbest/1`, crossover sipas satelitit dhe kontraktim
  radial të gjendjeve jashtë rajonit. U krahasuan F/CR fiks dhe përshtatje
  probabilistike të tyre. Të dyja janë shumë më të mira se DE bazë; përshtatja
  nuk tregoi epërsi të qartë në pilot dhe nuk është rekomandimi kryesor.
- **SAGE** kombinon 20% GA dhe 80% kërkim lokal. Kërkimi lokal perturbues përdor
  hapa 0.5/1/2/4 qeliza, kryesisht një satelit në një kohë. Pranon edhe lëvizje
  neutrale në pllaja, ndërsa kandidati global më i mirë ruhet veçmas. Pas
  stagnimit, zëvendëson një pjesë të satelitëve duke nisur nga kandidati më i
  mirë. Çdo provë dhe restart llogaritet në buxhet.
- **STM local** heq fazën GA nga SAGE (`genetic_fraction=0`): fillon me një
  formacion STM dhe shpenzon buxhetin në kërkimin lokal. Ky variant më i thjeshtë
  kishte medianën më të mirë për Intermediate dhe Hard në konfirmim, prandaj
  mori buxhetin shtesë. Hibridi mbetet si krahasim i riprodhueshëm, jo si
  kompleksitet i nevojshëm në konfigurimin e rekomanduar.

Crossover-i blend dha rezultate më të dobëta se shkëmbimi i satelitëve në pilot.
Përzierja 80% STM/20% uniform nuk tregoi përfitim të qëndrueshëm ndaj STM të
pastër. Këto ablacione ruhen për auditim; nuk pretendohen si përmirësime.
Nuk u krye kërkim shterues mbi të gjithë hiperparametrat e GA/DE.

## Eksperimentet dhe krahasimi i drejtë

| Studimi | Run-e | Buxheti/run | Seed-et | Vlerësime kërkimi |
|---|---:|---:|---|---:|
| Pilot, 8 variante × 3 instanca | 48 | 1,500 | 11, 22 | 72,000 |
| Konfirmim, 8 variante × 3 instanca | 72 | 10,000 | 101, 202, 303 | 720,000 |
| Zgjerim, STM local × 2 instanca | 6 | 50,000 | 404, 505, 606 | 300,000 |
| **Gjithsej optimizimi** | **126** | — | — | **1,092,000** |

Inicializimi, restart-et dhe gjeneratat e pjesshme numërohen brenda buxhetit.
Çdo run ka edhe një rivlerësim zyrtar të veçantë: 126 kontrolle shtesë. Studimi
i inicializimit, kontrollet përfundimtare, testet dhe prova paraprake me 300
kandidatë mbahen jashtë këtyre buxheteve. Asnjë kandidat i vjetër nuk u injektua
në run-et e reja. Zgjedhja për zgjerim dokumentohet te `extension-selection.json`.
Auditimi përfundimtar rilexoi edhe një herë të 126 vektorët nga JSON dhe i
rivlerësoi me ESA (126 thirrje të tjera jashtë kërkimit); të gjitha përputhen.

**Medianat me buxhet identik prej 10,000 vlerësimesh**, tri seed-e të reja:

| Metoda | Beginner | Intermediate | Hard |
|---|---:|---:|---:|
| `random-search` | -0.006802721 | -0.005353956 | -0.001784652 |
| `ga` | -0.020408163 | -0.274241523 | -0.012492564 |
| `de` | -0.006802721 | -0.005353956 | -0.001784652 |
| `stm-random` | -0.142857143 | -1.273646639 | -1.042831648 |
| `stm-ga` | -0.142857143 | -1.676977989 | -1.317668055 |
| `stm-de` | -0.142857143 | -1.593694230 | -1.268887567 |
| `sage` | -0.142857143 | -1.682926829 | -1.347412255 |
| `stm-local` | -0.142857143 | -1.693634741 | -1.352171327 |

SAGE arriti best-in më të mirë të konfirmimit për Intermediate, ndërsa STM local
kishte medianën më të mirë. Tri seed-e nuk mjaftojnë për pretendime të forta
statistikore mbi dallimet e vogla mes tyre. Përmirësimi ndaj baseline-it është
i dukshëm në të gjitha run-et e metodave të informuara. Tabelat JSON/CSV
përfshijnë best/median/mean/worst, devijimin standard të mostrës, kohën dhe
fraksionin e vlerësimeve me fitness zero.

Zgjerimi me STM local, 50,000 vlerësime/run dhe seed-e të reja:

| Instanca | Best | Medianë | Mesatare | Devijim standard | Sekonda mesatare |
|---|---:|---:|---:|---:|---:|
| Intermediate | -1.723378941 | -1.710291493 | -1.706325600 | 0.019343642 | 75.0 |
| Hard | -1.372397383 | -1.353361095 | -1.352171327 | 0.020846419 | 73.0 |

Rritja pesëfish e buxhetit nuk jep përmirësim pesëfish të cilësisë. Rezultatet
duhen lexuar bashkë me historikun: plateau-t dhe ndikimi i seed-it mbeten të
dukshme. Run-et e zgjerimit kanë seed-e të tjera, prandaj ndryshimi i medianës
nuk izolon vetëm efektin e buxhetit. Nuk vazhduam me buxhete pa kufi.

Kohët janë matje orientuese në një host Linux, jo benchmark i laptopit Windows.
Shuma e kohëve të raportuara të kërkimit është 1389.9 sekonda. Memoria RSS
maksimale e vëzhguar e procesit është 335.5 MiB; kjo është high-water mark i
procesit, përfshin evaluatorin/varësitë dhe nuk është matje e izoluar për algoritëm.
Mjedisi: Python 3.12.15, NumPy 1.26.4, SciPy 1.14.1, Matplotlib 3.9.2,
heyoka.py 3.2.0, pygmo 2.19.6. Pygmo nuk përdoret për të implementuar optimizerët.

## Pse Beginner është optimal

Me N satelitë, një projeksion mund të ketë maksimumi \(N(N-1)+1\) diferenca
të dallueshme: diferencat e renditura jo-zero dhe një zero. Për N=5 dhe G=11:

\[
F_{XY}+F_{XZ}+F_{YZ}\le
\frac{3(5\cdot4+1)}{(2\cdot11-1)^2}=\frac17.
\]

Prandaj fitness-i nuk mund të jetë më i vogël se −1/7. Kandidati i ruajtur e
arrin këtë në të tri vëzhgimet, pra arrin optimumin e kësaj instance. Për
Intermediate dhe Hard nuk pretendohet optimum global.

## Kompleksiteti dhe kostoja

Shënojmë me B buxhetin, N numrin e satelitëve, d=6, M=3, P popullatën dhe H
numrin e përmirësimeve të ruajtura. Përgatitja SVD ka kosto
\(O((d+3M)d^2+d^3)\), shumë të vogël këtu. Një transformim formacioni kushton
\(O(Nd^2)\). Le të jetë \(C_{ESA}\) kostoja e evaluatorit origjinal.

| Metoda | Kufiri kohor i kërkimit | Memoria ndihmëse e kërkimit |
|---|---|---|
| STM random / STM local | O(B(C_ESA + Nd²)) | O(Nd + H) |
| STM GA / STM DE | O(B(C_ESA + Nd² + P + log P)) | O(PNd + H) |
| SAGE | Shuma e dy fazave brenda të njëjtit B | O(PNd + H) në kulm |

Kufiri për popullatat është konservativ dhe përfshin zgjedhjen e prindërve/
donorëve dhe renditjen. H ≤ B. Evaluatori përdor gjithashtu rrjetë 3D O(G³)
dhe projeksione/korrelacione. Kostoja e tij përfshin
\(O(M(Nd^2+G^3+C_{corr}(G)))\); SciPy zgjedh automatikisht metodën e korrelacionit
(direkte O(G⁴) ose FFT O(G² log G)). Nuk kemi ndryshuar këtë zgjedhje.
Edhe një mutacion i vetëm sateliti kalon formacionin e plotë te ESA; nuk
pretendojmë vlerësim incremental O(1).

## Testet, origjinaliteti dhe skedarët

**126/126 teste kaluan**, përfshirë 89 të mëparshmet të pandryshuara dhe 37 të
reja. Testet mbulojnë kufijtë/STM/SVD, numërimin e thirrjeve, seed-et,
restart-et, ruajtjen e incumbent-it, historikun monoton, NaN/Inf, operatorët
e satelitëve, ruajtjen/rileximin e JSON-it, përputhjen me ESA dhe integrimin
nga Git bundle në linked worktree me skedarë lokalë të ndryshuar.

Implementimi i ri është i pavarur në Python; përdor ide standarde SVD, GA, DE
dhe local search të përshtatura për këtë problem. Kontributi është projektimi,
kombinimi, implementimi dhe vlerësimi i tyre, jo pretendimi se këto ide nuk
janë përdorur kurrë më parë. Nuk është kopjuar kod nga studentë ose optimizerë
të gatshëm. Burimet zyrtare ESA dhe licencat e tyre mbeten të pandryshuara.
Nuk kemi një korpus të projekteve të studentëve të tjerë për kontroll
ngjashmërie, prandaj nuk garantohet unikësi ndaj çdo projekti të mundshëm.
Commit-et, zgjedhjet konkrete algoritmike dhe eksperimentet japin prejardhje
të dokumentuar për punën individuale.

Grafikët PNG/PDF te `evidence/phase7/plots/` tregojnë konvergjencën, krahasimin
me buxhet të barabartë, konfigurimet 3D dhe projeksionet XY/XZ/YZ në të tri
kohët. Të gjithë përdorin të dhëna të ekzekutuara. Vizualizimi është i izoluar
nga evaluatori dhe vendimet e optimizimit.

Integrimi dhe komandat për Windows/PyCharm janë te **`PHASE7_WINDOWS.md`**.
Paketa përmban snapshot-in e projektit, evidencën, këtë raport dhe Git bundle-in
me commit-e të reja mbi bazën ekzistuese. Testet u ekzekutuan në Linux;
ekzekutimi në Windows duhet konfirmuar me komandat e dhëna. Dokumentimi
përfundimtar universitar mbetet për fazën e mëvonshme, sipas kërkesës.

Burime teknike: [kodi zyrtar ESA në revision-in e fiksuar](https://github.com/esa/SpOC3/tree/b45d23bb1950212510fde3d7b6159e9f9f6fbad4/Challenge%202%20Interferometric%20Mission/problems),
[kontrata SVD e NumPy 1.26](https://numpy.org/doc/1.26/reference/generated/numpy.linalg.svd.html),
dhe [ndryshimet e API-së heyoka.py](https://bluescarni.github.io/heyoka.py/breaking_changes.html).
Statistikat dhe prova e kufirit për Beginner janë nxjerrë nga kodi dhe
eksperimentet e këtij projekti; nuk janë rezultate të marra nga leaderboard-i.
