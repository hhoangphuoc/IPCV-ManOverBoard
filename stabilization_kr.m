clc, clear, close all

% Input video file which needs to be stabilized.
filename = "input video.MP4";
hVideoSource = VideoReader(filename);

outputVideo = VideoWriter('stabilized.mp4', 'MPEG-4');
outputVideo.FrameRate = hVideoSource.FrameRate;  % Match the frame rate of the input video
open(outputVideo);

hTM = vision.TemplateMatcher("ROIInputPort", true, ...
                            "BestMatchNeighborhoodOutputPort", true);

hVideoOut = vision.VideoPlayer("Name", "Video Stabilization");
hVideoOut.Position(1) = round(0.4*hVideoOut.Position(1));
hVideoOut.Position(2) = round(1.5*(hVideoOut.Position(2)));
hVideoOut.Position(3:4) = [650 350];

pos.template_orig = [918 448]; % [x y] upper left corner
pos.template_size = [100 60];   % [width height]
pos.search_border = [30 30];   % max horizontal and vertical displacement
pos.template_center = floor((pos.template_size-1)/2);
pos.template_center_pos = (pos.template_orig + pos.template_center - 1);
W = hVideoSource.Width; % Width in pixels
H = hVideoSource.Height; % Height in pixels
BorderCols = [1:pos.search_border(1)+4 W-pos.search_border(1)+4:W];
BorderRows = [1:pos.search_border(2)+4 H-pos.search_border(2)+4:H];
sz = [W, H];
TargetRowIndices = ...
  pos.template_orig(2)-1:pos.template_orig(2)+pos.template_size(2)-2;
TargetColIndices = ...
  pos.template_orig(1)-1:pos.template_orig(1)+pos.template_size(1)-2;
SearchRegion = pos.template_orig - pos.search_border - 1;
Offset = [0 0];
Target = zeros(60,100,3);  % Adjusted for color
firstTime = true;

while hasFrame(hVideoSource)
    input = im2double(readFrame(hVideoSource));

    % to find location of Target in the input video frame
    if firstTime
        Idx = int32(pos.template_center_pos);
        MotionVector = [0 0];
        firstTime = false;
    else
        IdxPrev = Idx;
        ROI = [SearchRegion, pos.template_size+2*pos.search_border];
        Idx = hTM(rgb2gray(input), rgb2gray(Target), ROI);  % Convert to grayscale for matching

        MotionVector = double(Idx-IdxPrev);
    end

    [Offset, SearchRegion] = updatesearch(sz, MotionVector, ...
        SearchRegion, Offset, pos);

    % Translate video frame to offset the camera motion
    Stabilized = imtranslate(input, Offset, "linear");
    Target = Stabilized(TargetRowIndices, TargetColIndices, :); % Adjust for color

    % adding black border for display
    Stabilized(:, BorderCols, :) = 0;
    Stabilized(BorderRows, :, :) = 0;

    TargetRect = [pos.template_orig-Offset, pos.template_size];
    SearchRegionRect = [SearchRegion, pos.template_size + 2*pos.search_border];

    % draw rectangles on input to show target and search region
    input = insertShape(input, "rectangle", [TargetRect; SearchRegionRect], ...
                        "Color", "white");
    
    % Display the offset (displacement) values on the input image
    txt = sprintf("(%+05.1f,%+05.1f)", Offset);
    input = insertText(input, [191 215], txt, "FontSize", 16, ...
                    "TextColor", "white", "BoxOpacity", 0);
    
    % Display video
    hVideoOut([input Stabilized]);
    writeVideo(outputVideo, Stabilized);
end

close(outputVideo);
disp('Video processing complete!');

function [Offset, SearchRegion] = updatesearch(sz, MotionVector, SearchRegion, Offset, pos)
    % Function to update Search Region for SAD and Offset for Translate
    A_i = Offset - MotionVector;
    AbsTemplate = pos.template_orig - A_i;
    SearchTopLeft = AbsTemplate - pos.search_border;
    SearchBottomRight = SearchTopLeft + (pos.template_size + 2*pos.search_border);

    inbounds = all([(SearchTopLeft >= [1 1]) (SearchBottomRight <= fliplr(sz))]);

    if inbounds
        Mv_out = MotionVector;
    else
        Mv_out = [0 0];
    end

    Offset = Offset - Mv_out;
    SearchRegion = SearchRegion + Mv_out;
end
