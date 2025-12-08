# TODO
___

1. Przyspieszenie treningu
- w `mecanum.sdf` ustawić parametr `<real_time_factor>1</real_time_factor>` na wartość **0** : `<real_time_factor>0</real_time_factor>`. <br> zniesie to ograniczenie symulacji do czasu rzeczywistego
- przeliczyć odpowiednio jak nalezy zmeinić parametr `TIME_STEP` w pliku `test_SB3.py`, tak aby dopasowac czas jednego time stampu do przyspieszenia symulacji

  
# Resolved 
___
1. render() - symulacja zgłasza warning że nie podano render_mode = True, ale można to zignorować - zaimplementowałem funkcję tak że tego nie potrzebuje. Przyszłościowo można dać ten render_mode jako argument i nic z nim nie ribić żeby nie krzyczało 

