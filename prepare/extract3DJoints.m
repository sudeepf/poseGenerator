function [ pose3 ] = extract3DJoints( p3d )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

        pose3 = zeros(14,3);
        
        %% 3D Pose Reading
        pose3(1, 2) = p3d(1,3*16-1);
        pose3(2, 2) = p3d(1,3*14-1);
        pose3(3, 2) = p3d(1,3*26-1);
        pose3(4, 2) = p3d(1,3*27-1);
        pose3(5, 2) = p3d(1,3*28-1);
        pose3(6, 2) = p3d(1,3*18-1);
        pose3(7, 2) = p3d(1,3*19-1);
        pose3(8, 2) = p3d(1,3*20-1);
        pose3(9, 2) = p3d(1,3*2-1);
        pose3(10, 2) = p3d(1,3*3-1);
        pose3(11, 2) = p3d(1,3*4-1);
        pose3(12, 2) = p3d(1,3*7-1);
        pose3(13, 2) = p3d(1,3*8-1);
        pose3(14, 2) = p3d(1,3*9-1);

        pose3(1, 1) = p3d(1,3*16-2);
        pose3(2, 1) = p3d(1,3*14-2);
        pose3(3, 1) = p3d(1,3*26-2);
        pose3(4, 1) = p3d(1,3*27-2);
        pose3(5, 1) = p3d(1,3*28-2);
        pose3(6, 1) = p3d(1,3*18-2);
        pose3(7, 1) = p3d(1,3*19-2);
        pose3(8, 1) = p3d(1,3*20-2);
        pose3(9, 1) = p3d(1,3*2-2);
        pose3(10, 1) = p3d(1,3*3-2);
        pose3(11, 1) = p3d(1,3*4-2);
        pose3(12, 1) = p3d(1,3*7-2);
        pose3(13, 1) = p3d(1,3*8-2);
        pose3(14, 1) = p3d(1,3*9-2);

        pose3(1, 3) = p3d(1,3*16);
        pose3(2, 3) = p3d(1,3*14);
        pose3(3, 3) = p3d(1,3*26);
        pose3(4, 3) = p3d(1,3*27);
        pose3(5, 3) = p3d(1,3*28);
        pose3(6, 3) = p3d(1,3*18);
        pose3(7, 3) = p3d(1,3*19);
        pose3(8, 3) = p3d(1,3*20);
        pose3(9, 3) = p3d(1,3*2);
        pose3(10, 3) = p3d(1,3*3);
        pose3(11, 3) = p3d(1,3*4);
        pose3(12, 3) = p3d(1,3*7);
        pose3(13, 3) = p3d(1,3*8);
        pose3(14, 3) = p3d(1,3*9);

end

