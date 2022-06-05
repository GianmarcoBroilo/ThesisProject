clear all
clc 
format long 
state = readtable('/Users/gianmarcobroilo/Desktop/ThesisResults/VLBI/5-0.5/initial_state.dat');
state = table2array(state);
x = state(1,8);
y = state(1,9);
z = state(1,10);

T = [-(x*z)/((1 - z^2/(x^2 + y^2 + z^2))^(1/2)*(x^2 + y^2 + z^2)^(3/2)),-(y*z)/((1 - z^2/(x^2 + y^2 + z^2))^(1/2)*(x^2 + y^2 + z^2)^(3/2)),(1/(x^2 + y^2 + z^2)^(1/2) - z^2/(x^2 + y^2 + z^2)^(3/2))/(1 - z^2/(x^2 + y^2 + z^2))^(1/2);
     -(2*y*(x/(x^2 + y^2)^(1/2) + 1))/((y^2/(x + (x^2 + y^2)^(1/2))^2 + 1)*(x + (x^2 + y^2)^(1/2))^2),(2*(1/(x + (x^2 + y^2)^(1/2)) - y^2/((x^2 + y^2)^(1/2)*(x + (x^2 + y^2)^(1/2))^2)))/(y^2/(x + (x^2 + y^2)^(1/2))^2 + 1),0];
