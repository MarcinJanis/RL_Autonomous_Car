# TODO
___
## 1. Przeprowadzić wstępny trening
- zapisać gdzieś wszystkie wyniki (csv i png) stworzone przez funkcję render() (chyba w fodlerze `training_logs` czy coś takiego)
- zapisać gdzieś najlepszy model
___
## 2. Napisać funkcję (nowy plik i w nowym folderze, zeby nie było to razem z programami do nauki):
- wczytuje model i wagi
- tworzy publishera na temacie /cmd_vel
- tworzy subscribera na tematach do kamery i lidaru
- podczytmuje noda, odbierając wejścia, przepuszczjąc przez sieć i wysyłąc komendy sterujące
- zapisuje gdzieś logi, dane to ewaluacji
- używajac funkcji z `trajectory_gt.py` rysuje gt i aktualną trajektorię w czasie rzeczywistym
- może rejestruje przebieg w postaci filmiku (?)
___
##3. Przyspieszenie treningu

W `mecanum.sdf` jest parametr `<real_time_factor>1</real_time_factor>`. Służy od określania prędkości symulacji.<br>
Wartość **1** oznacza synchronizację z rzeczywistym zegarem czyli 1 s w symulacji odpowiada 1 s w rzeczywistości. <br>
Wartość **0** oznacza symulajcę z największą prędkością - brak kontroli nad czasem wykonywania się. 

**Do zrobienia:**
- poczytać czy można określić np. real_time_factor = 2 żeby przyspieszyć symulację 2 krotnie.
- jeśli jest taka możliwość:
- w pliku `gazebo_car_env` dodać argument do kontruktora `_init_()` który przyjmuje liczbę jaką wpisaliśmy do `mecanum.sdf`.
- w funkcji `step()` jest pętla która odpowiada za czekanie, z wykorzystaniem funkcji z biblioteki `time`. Należy odpowiendnio przeskalować czas oczekiwania, tj. jeśli np. symulajca jest przyspieszona dwukrotnie, to czas musi być dwukrotnie mniejszy itp.
- w pliku `train_SB3` podać wszędzie gdzie trzeba ten argument
- przebadać jakie największe przyspieszenie można dać żeby sie wszystko nie wywaliło 
  


