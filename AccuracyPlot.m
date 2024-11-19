clc;
close all;
plot(info1.TrainingAccuracy)
hold on;
plot(info1.ValidationAccuracy)
grid on;
xlabel("Iterations")
ylabel("Accuracy")
legend("Training Accuracy","Validation Accuracy")
title("MyNet Accuracy Trained with ADAM")
figure(2)
plot(info2.TrainingAccuracy)
hold on;
plot(info2.ValidationAccuracy)
grid on;
xlabel("Iterations")
ylabel("Accuracy")
legend("Training Accuracy","Validation Accuracy")
title("AlexNet Accuracy Trained with ADAM")