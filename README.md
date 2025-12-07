___
Pierwsze testy

(uruchmomić z terminala z poziomu: ./RL_Autonomous_Car albo zmienic w pliku training_SB3.py ścieżki względne na bezwzględne)

Uruchomić plik:
./siapwpa_ros2_project-main_rl/autonomous_car/autonomous_car/check_env_SB3.py

jak to zadziała i nie będzie błędów to uruchomić:
./siapwpa_ros2_project-main_rl/autonomous_car/autonomous_car/training_SB3.py



___






# TODO 
___
# 1) Środowisko/ World do RL 
- Usunąć zbędnę pliki z projektu z SiAPWa, zrobić porządek
- Sprawdzić czy świat, mapa i wszystko jest dobrze skonfigurowane (narazie bez przeszkód)
- Konfiguracja czujników: 1x kamera + Lidar

# 2) Agent 
- Pojazd jako line follower, nauka w kilku fazach: <br>
  1 -> line follower, jazda za żółtą linią
  2 -> line follower, jazda po odpowiendim pasie
  3 -> line follower + unikanie przeszków
  
- Zaprojektować strategię karania i nagradzania, (narazie dla fazy pierwszej)
- Zaimplementować algorytm PPO 
- Zaimplementować architekturę sieci podjemującej decyzje **Kamera** + **Lidar** + **Prędkość**/**Prędkości kół** <br>
  **CNN (kamera) + MLP (lidar) -> Head (MLP)** <br> 



# 3) Wrapper
- Zaimplementować klasę wrappera, która jest zgodna ze standardem (niektórzy sugerują żeby zrobić klasę dziedziczącą np po wraperze od Gymnasium `gym.Env`)
- Wrapper ma być jednocześnie **Nodem** w Rosie. Ma pobierać stan auta, tj. **obraz**, **lidar**, **prędkość** (te dane będzie dostawać sieć do podejmowania decyzji), **pozycję** (do wykorzystania w karze/nagrodzie). <br> Ma wysyłać sygnał sterujący do symulacji (w przyszłości transformacja danych do symulacji na dane do rzeczywistego pojazdu tj. prędkość x + prędkość kątowa względm z -> prędkość x + skręt kół)
- Zadbać o synchronizację tj. żeby wrapper brał odpowiednie, aktualne klatki do obliczeń


___
# GIT INSTRUCTIONS
___
# 1) How to pull changes from orginal (upstream) repo:
- Check if upstream repo is followed: `git remote -v`
    origin and upstream repo shall be displayed 
- Pull changes from upstream repo: `git fetch upstream`
- Change for specific branch, eg. **develop** `git checkout develop`
- Merge changes from upstream repo to your branch `git merge upstream/develop`
- Push changes `git push origin develop`





