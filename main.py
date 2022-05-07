import numpy as np
import mido
import random
import music21
from typing import List, Tuple, Dict

# gammas for major and minor chords
major = [2, 2, 1, 2, 2, 2, 1]
minor = [2, 1, 2, 2, 1, 2, 2]

gammas = {"major": major, "minor": minor}

# mapping from note numbers to notes
notes_from_nums = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

# mapping from notes to their number
nums_from_notes = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}


def get_notes_with_octaves(midi_file: mido.MidiFile) -> (List[int], List[int]):
    # function to get notes from the initial midi file
    notes_ans = []
    octaves = []
    for x in midi_file.tracks[1]:
        if isinstance(x, mido.Message):
            if x.type == "note_on":
                notes_ans.append(x.note % 12)  # get note from midi track
                octaves.append(x.note // 12)  # get note's octave from midi track
    return notes_ans, octaves


def get_gamma(tone: str, start_note: int) -> List[int]:
    # function to get notes that will fit well for the note
    # from the gamma list with adding tone
    # gamma is different for major and minor tonic key
    # so, we should consider 2 cases

    gamma_ans = [start_note]
    if tone == "major":  # major case
        for i in major:
            start_note += i
            gamma_ans.append(start_note % 12)
    elif tone == "minor":  # minor case
        for i in minor:
            start_note += i
            gamma_ans.append(start_note % 12)
    return gamma_ans


def get_candidate_keys(notes: List[int]) -> Dict[Tuple[int, str], List[int]]:
    # determine keys, which fits in our gamma
    # these keys will be candidates for the tonic of the song
    # and after this we will determine which of these keys will fit best
    ret = {}
    for i in range(12):
        if set(notes).issubset(
            get_gamma("major", i)
        ):  # note i with major tonality will fit in the major gamma
            ret[(i, "major")] = get_gamma("major", i)

        if set(notes).issubset(
            get_gamma("minor", i)
        ):  # # note i with minor tonality will fit in the major gamma
            ret[(i, "minor")] = get_gamma("minor", i)

    return ret


def note_considering(
    notes_list: List[int],
    tonic_keys: Dict[Tuple[int, str], List[int]],
    note_type: str,
    key_scores: Dict[Tuple[int, str], int],
) -> Dict[Tuple[int, str], int]:
    # in this function we determine which of the candidate keys will fit best in the given melody
    # we use 2 metrics for this:
    # 1)considering last and first note and how will it fit in our current tonic key
    # 2)considering number of occurrences of similar notes
    # for this function we're considering first metric
    # if last/first note of the melody is exact equal to our tonic candidate we add 100 to aur metric
    # if last/first note of the melody is equal to mediant or dominant of our tonic candidate we add 50 to aur metric
    # if last/first note of the melody is equal to subdominant of our tonic candidate we add 25 to aur metric

    if note_type == "last":  # considering last note
        note = notes_list[-1]
    elif note_type == "first":  # considering first note
        note = notes_list[0]
    else:
        raise Exception("invalid note type")

    for key in tonic_keys:
        if key[0] == note:  # exact last note
            key_scores[key] = key_scores.get(key, 0) + 100

        if (
            tonic_keys[key][2] == note or tonic_keys[key][4] == note
        ):  # median or dominant of last note
            key_scores[key] = key_scores.get(key, 0) + 50

        if tonic_keys[key][3] == note:  # subdominant of last note
            key_scores[key] = key_scores.get(key, 0) + 25
    return key_scores


def repeatable_keys_considering(
    notes_list: [int], tonic_keys: {(int, str): [int]}, key_scores: {(int, str): int}
) -> {(int, str): int}:
    # in this function we're considering second metric of tonic key fitness
    # we chose key with most of the occurrences of notes and similar to it by gamma,
    # and then we add 25 to tonic key candidate with most of the occurrences of similar notes
    # we add only 25 because this criterion is less useful than considering last/first note of the melody

    max_val = 0  # max value of occurrences
    key_with_max_val = -1  # key that we will choose

    for key in tonic_keys:
        gamma = tonic_keys[key]
        considered_notes = [
            gamma[0],
            gamma[2],
            gamma[3],
            gamma[4],
        ]  # similar notes, which occurrences are we looking for
        val = 0
        for note in notes_list:
            if note in considered_notes:
                val += 1
        if val > max_val:  # updating max value of occurrences
            max_val = val
            key_with_max_val = key
    if key_with_max_val == -1:
        return key_scores
    else:
        key_scores[key_with_max_val] = (
            key_scores.get(key_with_max_val, 0) + 25
        )  # updating score of the chosen tonic key
        return key_scores


def choose_tonic_key(scores: {(int, str): int}) -> (int, str):
    # in this function we use the result of metric and according to them
    # chose tonic key which we will use further

    max_score = -1
    max_key = (0, 0)
    for key in scores:
        if scores[key] > max_score:
            max_score = scores[key]
            max_key = key
    return max_key


# triads for each type of chord
# adding this we will get chord which will sound better for the key
major_triad = [0, 4, 7]
minor_triad = [0, 3, 7]
dim_triad = [0, 3, 6]


class Chord:
    # class of chord
    # in this class we have next fields:
    # 1) octave - octave in which current chord situated
    # 2) chord_type - tone of the current chord (major/minor/diminished)
    # 3) first_chord_num - first note of the chord, this note determines name of the chord in the table
    # 4) plus_octave - list of bool values, which shows if this note after adding triad goes to the next octave
    # 5) chord_notes - list of notes used in this chord. We take mod of 12 value, to get note number
    def __init__(self, chord_type, chord_num, octave):
        self.octave = octave
        self.chord_type = chord_type
        self.first_chord_num = chord_num
        if chord_type == "major":
            self.chord_notes = [x + chord_num for x in major_triad]
        elif chord_type == "minor":
            self.chord_notes = [x + chord_num for x in minor_triad]
        elif chord_type == "dim":
            self.chord_notes = [x + chord_num for x in dim_triad]
        self.plus_octave = list(map(lambda x: x > 11, self.chord_notes))
        self.chord_notes = list(map(lambda x: x % 12, self.chord_notes))


def get_step_chords(tonic_key: Tuple[int, str], octave: int) -> List[Chord]:
    # in this function we chose chord according to table which will fit
    # for our chosen tonic key, according to the table given in the assignment
    # in further we will generate our accompaniment using only these step chords
    # because these chords will fit best for our tonic key of the given melody

    if (
        tonic_key[1] == "major"
    ):  # chose gamma according to which we will construct step chord
        chosen_gamma = major
    elif tonic_key[1] == "minor":
        chosen_gamma = minor
    else:
        raise Exception("something goes wrong")

    step_chords_for_key = []

    curr_key = nums_from_notes[tonic_key[0]]

    for i in range(
        len(chosen_gamma)
    ):  # construct step chords according to chosen gamma
        if chosen_gamma == major:
            if i == 0 or i == 3 or i == 4:
                step_chords_for_key.append(Chord("major", curr_key, octave))
            elif i == 1 or i == 2 or i == 5:
                step_chords_for_key.append(Chord("minor", curr_key, octave))
            else:
                step_chords_for_key.append(Chord("dim", curr_key, octave))
        elif chosen_gamma == minor:
            if i == 0 or i == 3 or i == 4:
                step_chords_for_key.append(Chord("minor", curr_key, octave))
            elif i == 2 or i == 5 or i == 6:
                step_chords_for_key.append(Chord("major", curr_key, octave))
            else:
                step_chords_for_key.append(Chord("dim", curr_key, octave))
        curr_key += chosen_gamma[i]
    return step_chords_for_key


def get_total_time(midi_file: mido.MidiFile) -> int:
    # function to get total time of the initial track
    # this function will be used to calculate number of chords
    # which we will use in the accompaniment

    time = 0
    for x in midi_file.tracks[1]:
        if isinstance(x, mido.Message):
            time += x.time
    return time


Genome = List[Chord]  # type of our genome used in genetic algorithm
Population = List[Genome]  # type of our population used in genetic algorithm


def generate_genome(length: int, step_chords: [Chord]) -> Genome:
    # function to generate genome
    # in this function we simply chose k random chords from the step chord list
    return random.choices(step_chords, k=length)


def generate_population(
    length: int, genome_length: int, step_chords: [Chord]
) -> Population:
    # function to generate population
    # we generate k genomes, and they are constructs population
    population = []
    for i in range(length):
        population.append(generate_genome(genome_length, step_chords))
    return population


def crossover(first_genome: Genome, second_genome: Genome) -> Tuple[Genome, Genome]:
    # function to crossover between 2 chosen genomes
    # we randomly generate number in range of length of the genome - p,
    # and then we form 2 new genomes which are combination of initial genomes
    # part from 0 to p of first genome with remaining part of second
    # and part from 0 to p of second genome with remaining part of first
    if len(first_genome) != len(second_genome):
        raise Exception("length of genomes should be the same")
    length = len(first_genome)
    if length < 2:
        return first_genome, second_genome

    cut_len = np.random.randint(1, length - 1)  # generate random number to cut genomes

    new_first_g = (
        first_genome[0:cut_len] + second_genome[cut_len:]
    )  # first resulted genome
    new_second_g = (
        second_genome[0:cut_len] + first_genome[cut_len:]
    )  # second resulted genome
    return new_first_g, new_second_g


def mutation(
    genome: Genome,
    step_chords: List[Chord],
    mutation_prob: float = 0.5,
    num_of_mutations: int = 10,
) -> Genome:
    # mutation function
    # we chose number of mutations to perform
    # and then generate 2 random variables
    # 1) position of the element in the genome
    # 2) probability to perform mutation
    # and if second variable is greater than given mutation probability
    # we replace genome chosen by the random variable by some random from step chord
    for x in range(num_of_mutations):
        random_genome_ind = np.random.randint(
            len(genome)
        )  # chose position in the genome
        prob = np.random.rand(1)  # chose current mutation probability
        if prob >= mutation_prob:
            genome[random_genome_ind] = np.random.choice(
                step_chords
            )  # replace chord in the genome
    return genome


# sequences of chord that will sound good
# we will use these sequences in the fitness function
# II-S-VI-T-III-D-VII
# 1) D-T
# 2) S-T
# 3) S-D-T
# 4) bad - D - S
# 5) It's bad to move into dominant decreasing side


def fitness_function(genome: Genome, step_chords: List[Chord]) -> int:
    # fitness function itself
    # in this function we determine how good our generated accompaniment will sound
    # we use rules shown above
    # and according to these rules we increase or decrease value of the result of the fitness function
    # and this function will return only 1 int number - goodness of our melody

    tonic_key = step_chords[0]
    subdominant = step_chords[3]
    dominant = step_chords[4]

    subdominant_group = [step_chords[5], subdominant, step_chords[1]]
    tonic_group = [step_chords[5], tonic_key, step_chords[2]]
    dominant_group = [step_chords[2], dominant, step_chords[6]]

    fitness = 0  # initial fitness

    if genome[0].chord_notes[0] in tonic_group:  # first chord in the tonic group - good
        fitness += 10

    for i in range(1, len(genome)):
        current = genome[i]
        previous = genome[i - 1]

        if current in subdominant_group and previous in dominant_group:  # D -> S is bad
            fitness -= 50
        if (
            current in dominant_group and previous in subdominant_group
        ):  # S -> D is good
            fitness += 25
        if current in tonic_group and (
            previous in subdominant_group
            or previous in dominant_group  # S -> T and D -> T is good
        ):
            fitness += 35
        if current == subdominant and previous == step_chords[1]:  # II -> S is bad
            fitness -= 20
        if current == step_chords[6] and previous == dominant:  # D -> VI is bad
            fitness -= 20
        if current == previous:  # repeatable chords is bad
            fitness -= 10

    return fitness


def selection(population: Population, step_chords: List[Chord]) -> Population:
    # this is selection function
    # in this function we chose 2 population to parents for further crossover
    # and the probability of being selected as parent is depends on its fitness score
    return random.choices(
        population,
        [fitness_function(genome, step_chords) for genome in population],
        k=2,
    )


def run_evolution(step_chords: List[Chord], chords_num) -> Population:
    # in this function we run our evolution
    # we firstly declare number of generation to choose the vest genome
    # then we generate population of genomes
    # and times equal generations number we generate next generation according to following algorithm:
    # * sort our population by fitness function result of the genomes
    # * take 2 best genomes
    # * and population size/2 do next:
    # * select 2 parents
    # * crossover them
    # * after this mutate them
    # * and append them to the next generation
    # and after this algorithm we will get next population with the same number of genomes
    # this new population should be better than previous in terms of fitness function of genomes
    generations_number = 100
    population = generate_population(100, chords_num, step_chords)

    for i in range(generations_number):
        sorted_population = sorted(
            population,
            key=lambda genome: fitness_function(genome, step_chords),
            reverse=True,
        )
        next_generation = sorted_population[0:2]  # select 2 genomes with best fitness

        for j in range(len(population) // 2 - 1):
            parents = selection(population, step_chords)  # selection
            first_offspring, second_offspring = crossover(
                parents[0], parents[1]
            )  # crossover selected parent
            first_offspring = mutation(
                first_offspring, step_chords
            )  # mutate these parent after crossover
            second_offspring = mutation(second_offspring, step_chords)
            next_generation += [first_offspring, second_offspring]

        population = next_generation
        print("step ", i, " completed")
    return population


def generate_output(
    midi_init: mido.MidiFile, chords_list: List[Chord], new_name: str
) -> None:
    # function that simply merges initial midi file
    # with our generated accompaniment and save new midi file
    # nothing interesting in terms of AI or music theory
    acc_mid = mido.MidiFile()
    track = mido.MidiTrack()
    acc_mid.tracks.append(track)
    time = 0
    ticks_per_beat = midi_init.ticks_per_beat

    track.append(mido.Message("program_change", program=12, time=0))

    for chord in chords_list:

        chord_notes = chord.chord_notes

        for note in chord_notes:
            track.append(mido.Message("note_on", note=int(note), velocity=50, time=0))
        track.append(
            mido.Message(
                "note_off", note=int(chord_notes[0]), velocity=50, time=ticks_per_beat
            )
        )
        track.append(
            mido.Message("note_off", note=int(chord_notes[1]), velocity=50, time=0)
        )
        track.append(
            mido.Message("note_off", note=int(chord_notes[2]), velocity=50, time=0)
        )

        track.append(mido.Message("note_off", note=chord_notes[1], velocity=0, time=0))
        track.append(mido.Message("note_off", note=chord_notes[2], velocity=0, time=0))

        time += ticks_per_beat

    merged_mid = mido.MidiFile()
    merged_mid.ticks_per_beat = ticks_per_beat

    merged_mid.tracks = midi_init.tracks + acc_mid.tracks
    merged_mid.save(f"{new_name}.mid")


def main():
    file = "input3.mid"
    midi = mido.MidiFile(file, clip=True)
    notes, octaves = get_notes_with_octaves(midi)
    ans = get_candidate_keys(notes)

    # calculation of score for tonic key decision
    scores = {}
    scores = note_considering(notes, ans, "last", scores)
    scores = note_considering(notes, ans, "first", scores)
    scores = repeatable_keys_considering(notes, ans, scores)

    score21 = music21.converter.parse(file)
    key21 = score21.analyze("key")
    tonic21 = (key21.tonic.name, key21.mode)

    tonic_key = choose_tonic_key(scores)
    tonic_key = (notes_from_nums[tonic_key[0]], tonic_key[1])

    if tonic21 != tonic_key:
        tonic_key = tonic21
    print(tonic_key)

    # calculation of octave of the initial melody and its time
    octave = sum(octaves) // len(octaves)
    step_chords = get_step_chords(tonic_key, octave)
    timestamp = 384
    chords_number = get_total_time(midi) // timestamp

    # execution of genetic algorithm
    final_population = run_evolution(step_chords, chords_number)
    genome_with_max_fitness = List[Chord]
    max_fitness = -1000

    # choosing of the best genome from the final population
    for genome in final_population:
        cur_fit = fitness_function(genome, step_chords)
        if cur_fit > max_fitness:
            max_fitness = cur_fit
            genome_with_max_fitness = genome

    # accompaniment of the melody should be 2 octaves lower
    for chord in step_chords:
        for i in range(len(chord.chord_notes)):
            chord.chord_notes[i] += ((octave - 2) + chord.plus_octave[i]) * 12

    # generate resulted midi file
    generate_output(midi, genome_with_max_fitness, "output")


if __name__ == "__main__":
    main()
