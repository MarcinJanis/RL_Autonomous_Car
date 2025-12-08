# TODO
___

1. Przyspieszenie treningu
- w `mecanum.sdf` ustawić parametr `<real_time_factor>1</real_time_factor>` na wartość **0** : `<real_time_factor>0</real_time_factor>`. <br> zniesie to ograniczenie symulacji do czasu rzeczywistego
- przeliczyć odpowiednio jak nalezy zmeinić parametr `TIME_STEP` w pliku `test_SB3.py`, tak aby dopasowac czas jednego time stampu do przyspieszenia symulacji

2. Dopasowac wartosci kary:

przykładowe warotści z jakiegoś przejazdu przy aktualnych wartościach kary
> Episode 3 finished with 752 steps.
> Rewards: 
> velocity: 406.1731872558594 
> trajectory: -6856.5400390625 
> ang_vel: -35.17915725708008 
> collision: -15.0 
> timeout: 0.0

- zwiększyć znacznie karę za kolizję, lub dac sto razy mniejszą za trajektorię, z 15 - 20 razy mniejszą za prędkość kątową, i 100 razy mniejszą za pręsdkość liniwoą
  
# Resolved 
___
1. render() - symulacja zgłasza warning że nie podano render_mode = True, ale można to zignorować - zaimplementowałem funkcję tak że tego nie potrzebuje. Przyszłościowo można dać ten render_mode jako argument i nic z nim nie ribić żeby nie krzyczało 

