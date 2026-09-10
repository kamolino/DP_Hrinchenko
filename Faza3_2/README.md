# Fáza 3.2 — Dosahovanie cieľa robotom Franka FR3 pomocou Double DQN

V tejto fáze som sa zameral na presné dosahovanie cieľovej polohy XYZ a na overenie riadenia v širšej pracovnej oblasti robota. Použil som model sedemosového robota Franka FR3 v simulátore MuJoCo.

Výsledné riešenie spája **plánovanie cesty a Double DQN**. Plánovač vypočíta prípustné medzipolohy a naučená sieť vyberá pohybové príkazy. Ide teda o hybridný systém; samotný DQN neplánuje celú cestu k cieľu.

## Výsledky

Finálny test obsahuje 216 úloh, ktoré sa nepoužili na výber modelu: 72 globálnych cieľov z domácej polohy, 72 hraničných cieľov a 72 úloh s rôznymi štartovacími polohami.

| Metóda na rovnakých 216 úlohách | Úspešné úlohy | Priemerná konečná chyba | 95. percentil chyby | Kolízie |
|---|---:|---:|---:|---:|
| Lokálny DQN bez globálneho plánovača | 100/216 (46,30 %) | 319,302 mm | 1323,715 mm | 10 |
| Plánovač + kĺbový Double DQN | **215/216 (99,54 %)** | **0,220 mm** | **0,469 mm** | **0** |

Za úspech považujem vzdialenosť TCP od cieľa najviac **1 mm**, pri rýchlosti najviac **5 mm/s**, nepretržite počas **0,3 s**. TCP je referenčný bod na konci ramena. Podmienky sa kontrolujú pri každom fyzikálnom kroku 2 ms; sieť vyberá akciu každých 50 ms. Časový limit úlohy je 30 s simulácie.

Jediná neúspešná úloha skončila po časovom limite s chybou 1,051 mm. Je zahrnutá v tabuľke aj vo vizualizácii. Samostatná validačná sada dosiahla 72/72 úspechov. Porovnanie hodnotí celé riadiace systémy, nie izolovaný vplyv zmeny algoritmu DQN.

![Vyhodnotenie pracovnej oblasti](results/workspace/visualization/workspace_coverage.png)

## Ako systém funguje

1. **Zadanie cieľa:** vstupom je aktuálna kĺbová poloha a cieľ XYZ v metroch.
2. **Výpočet cesty:** numerická inverzná kinematika nájde kĺbovú polohu pre cieľ. Plánovač skontroluje priamu cestu v kĺbovom priestore. Ak je zablokovaná, môže použiť RRT-Connect. Vo finálnom teste stačila priama cesta k vhodnému IK riešeniu vo všetkých 216 úlohách; RRT sa overuje samostatným testom.
3. **Riadenie pomocou Double DQN:** cesta sa rozdelí na medzipolohy. Sieť vyberá z 15 akcií: držanie alebo kladný/záporný pohyb jedného zo siedmich kĺbov.
4. **Kontrola pohybu:** geometrická kontrola odmietne kolízny príkaz. Pohyb prebieha cez fyziku MuJoCo. Dosiahnutie cieľa sa vyhodnocuje z polohy a rýchlosti TCP.

Sieť používa 12 príznakov na akciu a vrstvy **12 → 64 → 64 → 1**. Príznaky opisujú chybu medzipolohy, rýchlosť a rozdiel medzi príkazom a stavom kĺbu. Pri tréningu som použil demonštrácie simulovaného učiteľa: 6744 prechodov zo 160 epizód, 2500 prípravných aktualizácií a 2500 offline Double DQN aktualizácií s regularizáciou podľa demonštrácií. Pri vykonávaní sa učiteľ nepoužíva. Double DQN oddeľuje výber nasledujúcej akcie od vyhodnotenia jej hodnoty pomocou cieľovej siete.

## Rozsah overenia

Ciele sú rozdelené do 12 smerových sektorov po 30°, s rôznymi vzdialenosťami, výškami a štartmi. Polomer cieľov od osi základne je približne 0,018–0,855 m a výška 0,034–1,186 m. Platí rezerva 0,08 rad od kĺbových limitov a výška TCP aspoň 3 cm nad podlahou. **Výška 3 cm je prevádzkové obmedzenie, nie tolerancia dosiahnutia cieľa.**

Dosiahnuteľnosť cieľovej polohy je doložená kĺbovou konfiguráciou `witness_q` v testovacích dátach. Regulátor ju nedostáva; používa iba štart a cieľ XYZ. Výsledok 99,54 % platí pre uloženú konečnú vzorku, nie ako záruka dosiahnutia každého bodu spojitého priestoru. Podrobná metodika je v [BENCHMARK.md](results/workspace/BENCHMARK.md).

## Vizualizácia

- [Interaktívna pracovná oblasť](results/workspace/visualization/workspace_viewer.html): po stiahnutí otvorte HTML v prehliadači. Obsahuje všetkých 216 cieľov a 23 vybraných trajektórií vrátane neúspešnej; umožňuje prepínať projekciu a prehrávať pohyb. GitHub zobrazuje HTML ako zdrojový súbor.
- [Animácia robota v MuJoCo](results/workspace/visualization/robot_replay.gif): najdlhšia úspešná zaznamenaná trajektória. Animácia je zrýchlená a predstavuje prehratie merania, nie nové meranie. Zväčšená zelená značka neznázorňuje toleranciu 1 mm.
- [Mapa výsledkov](results/workspace/visualization/workspace_coverage.png): všetky testované ciele, úspešnosť podľa smeru a rozdelenie chýb.

## Spustenie

Príkazy spúšťajte **z koreňového priečinka repozitára**. Overenie prebehlo na Linuxe s Python 3.12.3, MuJoCo 3.10.0, PyTorch 2.13.0, NumPy 2.5.1, Matplotlib 3.11.1 a Pillow 12.3.0. Výpočet používa CPU; GPU nie je potrebné. `requirements.txt` uvádza minimálne požiadavky, nie presne uzamknuté verzie.

```bash
python3 -m venv Faza3_2/.venv
Faza3_2/.venv/bin/python -m pip install -r Faza3_2/requirements.txt

# Ukážka 12 uložených úspešných úloh v okne MuJoCo.
bash Faza3_2/run_workspace_demo.sh

# Vlastný cieľ XYZ v metroch.
Faza3_2/.venv/bin/python Faza3_2/code/run_workspace_dqn.py \
  --goal -0.45 0.25 0.55 --viewer
```

Model sa načíta automaticky; nové učenie pred ukážkou nie je potrebné. Demo je prezentačný výber, nie test úspešnosti. Okno vyžaduje grafické prostredie; pre výpočet bez okna vynechajte `--viewer`. Medzi úlohami sa simulácia resetuje na štartovaciu polohu, tento reset nie je fyzickým pohybom. Nedosiahnutý cieľ sa vykáže ako neúspech.

```bash
# Automatické kontroly.
OMP_NUM_THREADS=1 Faza3_2/.venv/bin/python -m unittest discover -s Faza3_2/tests -v

# Opakovanie celého testu bez okna, s novým názvom výstupu.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
Faza3_2/.venv/bin/python Faza3_2/code/run_workspace_dqn.py \
  --tasks Faza3_2/results/workspace/test_tasks.json \
  --output Faza3_2/results/workspace/hybrid_test_repeat.json

# Nový tréning do samostatného priečinka.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
Faza3_2/.venv/bin/python Faza3_2/code/train_joint_waypoint_dqn.py \
  --output Faza3_2/results/workspace/joint_run_002
```

## Obsah

| Priečinok | Obsah |
|---|---|
| `code/` | Prostredie, Double DQN, plánovač, tréning, hodnotenie a vizualizácia |
| `mujoco/franka_fr3/` | Model robota, mesh súbory a pôvodná licencia |
| `tests/` | Kontroly prostredia, siete, plánovača a testovacích úloh |
| `results/workspace/joint_run_001/` | Natrénovaný Double DQN a záznam tréningu |
| `results/workspace/` | Testovacie úlohy, validačné a finálne merania, porovnanie |
| `results/workspace/visualization/` | Graf, HTML a animácia |
| `results/precision/run_001/best.pt` | Lokálny model použitý v porovnaní; má iný vstupný formát |

Číselné výsledky a trajektórie sú zachované; cesty v textových metadátach sú relatívne ku koreňu repozitára. Virtuálne prostredie a pomocné pracovné experimenty nie sú súčasťou priečinka.

## Obmedzenia

Zatiaľ ide o výsledok v simulácii. Používa sa presná kinematika a ideálna kompenzácia gravitácie, bez šumu snímačov, komunikačného oneskorenia a záťaže nástroja. Cieľ určuje XYZ, nie orientáciu uchopovača. Nové vonkajšie prekážky neboli súčasťou meraného benchmarku.

Presnosť simulácie nie je meraním presnosti ani opakovateľnosti fyzického FR3. Prenos na reálny robot vyžaduje samostatné rozhranie riadenia, kalibráciu a overenie dynamických limitov. Tieto skripty sa k fyzickému robotu nepripájajú.

Model pochádza z MuJoCo Menagerie / Franka Description. [Pôvodná licencia](mujoco/franka_fr3/LICENSE) je zachovaná; [úpravy modelu](mujoco/franka_fr3/PROJECT_MODIFICATIONS.md) sú uvedené samostatne.
