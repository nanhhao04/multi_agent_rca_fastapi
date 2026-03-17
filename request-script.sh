TIMES=10

for ((i=1;i<=TIMES;i++))
do
    echo "Round $i"

    for j in {1..10}; do curl http://localhost:8000/; done
    for j in {1..10}; do curl http://localhost:8000/io_task; done
    for j in {1..10}; do curl http://localhost:8000/cpu_task; done
    for j in {1..10}; do curl http://localhost:8000/random_sleep; done
    for j in {1..10}; do curl http://localhost:8000/random_status; done
    for j in {1..10}; do curl http://localhost:8000/chain; done
    curl http://localhost:8000/error_test

    sleep 5
done