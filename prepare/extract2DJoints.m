function [ pose2 ] = extract2DJoints( p2d )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    
    pose2 = zeros(14,2);
    j=1;
    pose2(1, 1) = p2d(j,2*16-1);
    pose2(2, 1) = p2d(j,2*14-1);
    pose2(3, 1) = p2d(j,2*26-1);
    pose2(4, 1) = p2d(j,2*27-1);
    pose2(5, 1) = p2d(j,2*28-1);
    pose2(6, 1) = p2d(j,2*18-1);
    pose2(7, 1) = p2d(j,2*19-1);
    pose2(8, 1) = p2d(j,2*20-1);
    pose2(9, 1) = p2d(j,2*2-1);
    pose2(10, 1) = p2d(j,2*3-1);
    pose2(11, 1) = p2d(j,2*4-1);
    pose2(12, 1) = p2d(j,2*7-1);
    pose2(13, 1) = p2d(j,2*8-1);
    pose2(14, 1) = p2d(j,2*9-1);


    pose2(1, 2) = p2d(j,2*16);
    pose2(2, 2) = p2d(j,2*14);
    pose2(3, 2) = p2d(j,2*26);
    pose2(4, 2) = p2d(j,2*27);
    pose2(5, 2) = p2d(j,2*28);
    pose2(6, 2) = p2d(j,2*18);
    pose2(7, 2) = p2d(j,2*19);
    pose2(8, 2) = p2d(j,2*20);
    pose2(9, 2) = p2d(j,2*2);
    pose2(10, 2) = p2d(j,2*3);
    pose2(11, 2) = p2d(j,2*4);
    pose2(12, 2) = p2d(j,2*7);
    pose2(13, 2) = p2d(j,2*8);
    pose2(14, 2) = p2d(j,2*9);

end

