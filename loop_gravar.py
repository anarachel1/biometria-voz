from loop_listen.loop_listen  import Loop_listen as Loop

i=3  #quantidade de vezes você quer gravar o áudio
try:
    for i in range(i):
        audio = Loop(filename=str(i), threshold=False, max_seconds=3) # filename = nome do arquivo, threshold = abertura do microfone, max_seconds = tempo maximo de cada gravação
        audio.listen()
except KeyboardInterrupt:
    print('interrupted!')

print('Finalizado com sucesso')