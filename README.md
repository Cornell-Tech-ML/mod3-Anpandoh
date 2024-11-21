# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## 100 Hidden Layers
### Split
11 min 36 second
Epoch  0  loss  4.974794323511812 correct 32
Epoch  10  loss  4.866075460767081 correct 42
Epoch  20  loss  4.452935023908912 correct 39
Epoch  30  loss  3.1204575954818154 correct 45
Epoch  40  loss  3.0979522578514453 correct 45
Epoch  50  loss  3.6281284983105166 correct 43
Epoch  60  loss  2.0102994763035102 correct 49
Epoch  70  loss  1.4664962610995271 correct 45
Epoch  80  loss  2.502620995539208 correct 49
Epoch  90  loss  1.9281485980573676 correct 47
Epoch  100  loss  2.1940869274636503 correct 48
Epoch  110  loss  1.4787541646757865 correct 48
Epoch  120  loss  1.263647801778647 correct 46
Epoch  130  loss  1.56677790718257 correct 48
Epoch  140  loss  1.541647698848383 correct 48
Epoch  150  loss  0.6923766414914844 correct 48
Epoch  160  loss  0.6565255024502188 correct 50
Epoch  170  loss  2.2846881030895165 correct 49
Epoch  180  loss  1.0463522898450472 correct 47
Epoch  190  loss  1.7577502943582035 correct 47
Epoch  200  loss  0.48397059958641936 correct 49
Epoch  210  loss  0.7976809402579368 correct 50
Epoch  220  loss  1.578133567601871 correct 48
Epoch  230  loss  1.3979002091224177 correct 48
Epoch  240  loss  0.8768089165576034 correct 50
Epoch  250  loss  1.3648225503378946 correct 49
Epoch  260  loss  1.0507172044004411 correct 49
Epoch  270  loss  0.5174502356925563 correct 49
Epoch  280  loss  2.6861209662365777 correct 46
Epoch  290  loss  0.4929772493288891 correct 50
Epoch  300  loss  0.6117936196462845 correct 50
Epoch  310  loss  0.7393048548177391 correct 49
Epoch  320  loss  0.21943784103150327 correct 48
Epoch  330  loss  0.6260285714803779 correct 49
Epoch  340  loss  1.0187334353429216 correct 50
Epoch  350  loss  1.2429499913593505 correct 50
Epoch  360  loss  0.603376963453909 correct 48
Epoch  370  loss  0.18490361660506321 correct 49
Epoch  380  loss  0.8290414637159247 correct 49
Epoch  390  loss  0.14173384587725787 correct 49
Epoch  400  loss  0.49109535557611866 correct 49
Epoch  410  loss  0.23736286859093544 correct 50
Epoch  420  loss  1.070753109734983 correct 50
Epoch  430  loss  1.3995939776911388 correct 49
Epoch  440  loss  0.04876306614577331 correct 50
Epoch  450  loss  1.612705599942421 correct 49
Epoch  460  loss  0.37252833317744816 correct 49
Epoch  470  loss  0.033039545749305836 correct 50
Epoch  480  loss  0.35616778821366085 correct 50
Epoch  490  loss  0.5197568422011701 correct 49

### XOR

11 min 34 seconds
Epoch  0  loss  6.764605472673965 correct 34
Epoch  10  loss  5.913263957041755 correct 41
Epoch  20  loss  5.204119858833385 correct 42
Epoch  30  loss  3.175175500679197 correct 43
Epoch  40  loss  5.386119967846439 correct 46
Epoch  50  loss  2.7435624847807376 correct 44
Epoch  60  loss  2.895999756358454 correct 44
Epoch  70  loss  2.9135403565622813 correct 44
Epoch  80  loss  1.6710988001173146 correct 44
Epoch  90  loss  1.775379186641775 correct 48
Epoch  100  loss  3.122077413150103 correct 48
Epoch  110  loss  1.666772848068336 correct 49
Epoch  120  loss  0.9362277398004752 correct 49
Epoch  130  loss  2.6170149219613625 correct 47
Epoch  140  loss  2.673351174159797 correct 48
Epoch  150  loss  2.727446354823298 correct 49
Epoch  160  loss  1.4992308194328485 correct 49
Epoch  170  loss  1.7208755675874863 correct 50
Epoch  180  loss  1.8014910768175856 correct 50
Epoch  190  loss  1.266328684600353 correct 50
Epoch  200  loss  1.917086543766721 correct 50
Epoch  210  loss  1.3014378219279898 correct 50
Epoch  220  loss  1.7460568070681566 correct 50
Epoch  230  loss  1.1207904382415648 correct 50
Epoch  240  loss  0.43796745139727367 correct 50
Epoch  250  loss  0.6107759072100943 correct 49
Epoch  260  loss  0.7247489804464256 correct 50
Epoch  270  loss  1.0011055516952487 correct 50
Epoch  280  loss  0.9408964477362269 correct 49
Epoch  290  loss  1.7117006623882278 correct 49
Epoch  300  loss  1.142124225085328 correct 50
Epoch  310  loss  0.39475577441226745 correct 50
Epoch  320  loss  0.9960288157252686 correct 50
Epoch  330  loss  0.5642312483765679 correct 50
Epoch  340  loss  0.27383139631746733 correct 49
Epoch  350  loss  1.293975300829945 correct 50
Epoch  360  loss  0.9924389965248838 correct 50
Epoch  370  loss  0.6313809729328972 correct 50
Epoch  380  loss  0.7440098651775249 correct 50
Epoch  390  loss  0.28599232985904166 correct 50
Epoch  400  loss  0.18631168284904503 correct 50
Epoch  410  loss  0.6122520492166857 correct 50
Epoch  420  loss  0.3782444924130708 correct 50
Epoch  430  loss  0.2534213944262367 correct 50
Epoch  440  loss  0.34481598098528504 correct 50
Epoch  450  loss  0.3972605242870362 correct 50
Epoch  460  loss  0.3218059497901668 correct 50
Epoch  470  loss  0.34762023065924696 correct 50
Epoch  480  loss  0.1535732282802127 correct 50
Epoch  490  loss  0.09017544670889063 correct 50