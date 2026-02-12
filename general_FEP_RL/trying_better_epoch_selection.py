#%% 
def resample_even(l, excess_epoch_numbers):
    return([epoch_number for epoch_number in l if not epoch_number in excess_epoch_numbers])



max_length = 8



def get_excess_epoch_numbers(epoch_numbers):
    if len(epoch_numbers) <= max_length:
        return None
    epoch_number_indeces = {i : e for i, e in enumerate(epoch_numbers)}
    excess_indeces = []
    differences = [e2 - e1 for e1, e2 in zip(epoch_numbers, epoch_numbers[1:])]
    print("differences:", differences)
    indeces_of_smallest_differences = [i for i, e in enumerate(differences) if e == min(differences)]
    print("indeces_of_smallest_differences", indeces_of_smallest_differences)
    while len(epoch_numbers) - len(excess_indeces) > max_length:
        print(f"lengths: {len(epoch_numbers)}, {len(excess_indeces)}")
        step = 0
        index = len(epoch_numbers) // 2
        while index not in indeces_of_smallest_differences or index in excess_indeces:
            index =+ steps 
            if step > 0:
                step += 1 
            if step < 0:
                step -= 1
            step *= -1
        print("index:", index)
        excess_indeces.append(index)
    print(excess_indeces)
    return [epoch_number_indeces[index] for index in excess_indeces]
    


epoch_numbers = [] 
for i in range(16):
    print()
    epoch_numbers.append(i)
    excess_epoch_numbers = get_excess_epoch_numbers(epoch_numbers)
    if excess_epoch_numbers is not None:
        epoch_numbers = resample_even(epoch_numbers, excess_epoch_numbers)
    print("epoch_numbers:", epoch_numbers)


print("\n\n")
print([])