function maxMinAvg(arr) {
    var max = arr[0];
    var min = arr[0];
    var sum = arr[0]; //changed from original post
    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
        if (arr[i] < min) {
            min = arr[i];
        }
        sum = sum + arr[i];
    }
    return [max, min, sum/arr.length]; //changed from original post
}

$(function(){
    $("button[name=retreive_water_depth]").on('click', function(e){
        console.log("processing depth");
        e.preventDefault(); 

        var mean = 100;
        var sd = 15;
        var num_samples = 30;

        var min = 9999999;
        var max = -9999999;
        var sum = 0;

        for (var i = 0; i < num_samples; i++) { 
            var val = Math.random() * sd + mean;
            if (val < min){
                min = val;
            }
            if (val > max){
                max = val;
            }
            sum += val;
        }

        min = Math.round(min);
        max = Math.round(max);
        var avg = Math.round(sum / num_samples);

        $('input[name=min_depth]').val(min);
        $('input[name=max_depth]').val(max);
        $('input[name=avg_depth]').val(avg);
        
    });

});
