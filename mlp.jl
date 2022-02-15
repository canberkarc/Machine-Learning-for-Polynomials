using Flux: @epochs, throttle, onecold
using Flux
using Base
using BSON: @save, @load
using Statistics

function input(f1,f2)
	train = []
	lines = readlines(f1)
	
	for line in lines
		line_splitted = split(line, " ")	
	    i_arr = []
    	check = false
	    for num in line_splitted
	    	if string(num) == "\"" || string(num) == " "
	    		continue
	    	elseif string(num) == "-"
	    		check = true
	    	elseif check == true
	    		val = Base.parse(Float64, string(num)) * -1
	        	push!(i_arr, val)
	        	check = false
	        	continue		
	    	else
	    		val = Base.parse(Float64, string(num))
	        	push!(i_arr, val)
	    	end
	    end
		i_arr = convert(Array{Float32},i_arr) 
		i_arr = reshape(i_arr,:,1)
		push!(train, i_arr)
	end

	train = convert(Array{Array{Float32}},train)
	train = reshape(train,1,:)

	lines = Tuple(readlines(f2))
	test = []

	for i in lines
		if tryparse(Float64,string(i)) !== nothing
			val = parse(Float64,string(i))
			push!(test, val)
		end
	end

	test = convert(Array{Float32}, test) 
	test = reshape(test,1,:)
	return train, test

end

function input_test(f1,f2)
	train = []
	lines = readlines(f1)
	
	for line in lines
		line_splitted = split(line, " ")	
	    i_arr = []
    	check = false
	    for num in line_splitted
	    	if string(num) == "\"" || string(num) == " "
	    		continue
	    	elseif string(num) == "-"
	    		check = true
	    	elseif check == true
	    		val = Base.parse(Float64, string(num)) * -1
	        	push!(i_arr, val)
	        	check = false
	        	continue		
	    	else
	    		val = Base.parse(Float64, string(num))
	        	push!(i_arr, val)
	    	end
	    end
		i_arr = convert(Array{Float32},i_arr) 
		i_arr = reshape(i_arr,:,1)
		push!(train, i_arr)
	end

	train = convert(Array{Array{Float32}},train)

	lines = Tuple(readlines(f2))
	test = []

	for i in lines
		if tryparse(Float64,string(i)) !== nothing
			val = parse(Float64,string(i))
			push!(test, val)
		end
	end

	test = convert(Array{Float32}, test) 
	test = reshape(test, 1,:)
	return train, test

end

function MLP_model(N,num_of_classes)
	model = Chain(Dense(N,100),
		        Dropout(0.2),
		        Dense(100,200),
		        Dropout(0.1),
		        Dense(200,num_of_classes,sigmoid)
		        )
	return model
end

function eval_model(x, model)
	x = reshape(x,1,:)
	output = [model(k) for k in x][end]
    output
end

function main()
	num_of_classes = 100
	num_epochs = 50
	x_train_path = "/home/canberk/Desktop/coeffs_train.txt"
	y_train_path = "/home/canberk/Desktop/roots_n_train.txt"

	x_train, y_train = input(x_train_path,y_train_path)

	x_test_path = "/home/canberk/Desktop/coeffs_test.txt"
	y_test_path = "/home/canberk/Desktop/roots_n_test.txt"

	x_test, y_test = input_test(x_test_path,y_test_path)

	loss(x, y) = sum(Flux.Losses.binarycrossentropy(eval_model(x,model), y))

	# TRAINING
	# Please uncomment to train
	#=
	model = MLP_model(100,num_of_classes)
	ps = Flux.params(model)

	opt = Flux.ADAM(0.001)

    @info("Training...")
 	# callback function during training
    evalcb() = @show(sum(loss.(x_test, y_test)))
	
	# Save weights of trained model
	@epochs num_epochs Flux.train!(loss, ps, [(x_train, y_train)], opt)
	@save "mlp_model.bson" model
	=#
	
	@load "mlp_model.bson" model
	# after training, evaluate the loss
	println("Test loss: ", sum(loss(x_test, y_test)))
	# Test loss: -55.798344
end

main()
