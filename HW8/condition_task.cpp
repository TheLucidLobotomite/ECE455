#include "taskflow/taskflow/taskflow.hpp"


int main() {
    tf::Executor executor;
    tf::Taskflow taskflow("Condition Task Demo");

    int counter = 0;
    const int limit = 5;

    // Initialization task
    auto init = taskflow.emplace([&]() {
        printf("Initialize counter = %d\n", counter);
    });

    // Loop task with a condition
    auto loop = taskflow.emplace([&]() {
        printf("Loop iteration %d\n", counter);
        counter++;
        return (counter < limit) ? 0 : 1;  // 0 => go back, 1 => exit
    }).condition();

    // Completion task
    auto done = taskflow.emplace([]() {
        printf("Loop done.\n");
    });

    // Define dependencies
    init.precede(loop);
    loop.precede(loop, done);  // self-edge enables iteration

    executor.run(taskflow).wait();

    return 0;
}
