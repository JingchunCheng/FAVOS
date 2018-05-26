function [J, inters, fp, fn] = jaccard_single( object, ground_truth )

% Make sure they're binary
object       = logical(object);
ground_truth = logical(ground_truth);

% Intersection between all sets
inters = object.*ground_truth;
fp     = object.*(1-inters);
fn     = ground_truth.*(1-inters);

% Areas of the intersections
inters = sum(inters(:)); % Intersection
fp     = sum(fp(:)); % False positives
fn     = sum(fn(:)); % False negatives

% Compute the fraction
denom = inters + fp + fn;
if denom==0
    J = 1;
else
    J =  inters/denom;
end
end
