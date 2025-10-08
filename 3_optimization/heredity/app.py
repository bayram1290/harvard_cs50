# Assess the likelihood that a person will have a particular genetic trait

import itertools
import csv
import sys

PROBS = {

    "gene": { # Unconditional probabilities for having gene
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {
        2: { # Probability of trait given two copies of gene
            True: 0.65,
            False: 0.35
        },
        1: { # Probability of trait given one copy of gene
            True: 0.56,
            False: 0.44
        },
        0: { # Probability of trait given no gene
            True: 0.01,
            False: 0.99
        }
    },
    "mutation": 0.01 # Mutation probability
}


def load_data(file_path):
    """
    Load gene and trait data from a file into a dictionary.

    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.

    Returns:
        dict: A dictionary where each key is a name and each value is a dictionary
            containing the corresponding information from the CSV file.
    """
    data = dict()

    with open(file_path, encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = row['name']
            data[name] = {
                'name': name,
                'mother': row['mother'],
                'father': row['father'],
                'trait': (True if row['trait'] == '1' else False if row['trait'] == '0' else None)
            }

    return data

def power_set(inputs: set) -> list:
    """
    Return a list of all subsets of the input set.

    Args:
        inputs: set: The input set

    Returns:
        list: A list of all subsets of the input set
    """
    inputs = list(inputs)

    if len(inputs) > 0:
        return [
            set(inputs) for inputs in itertools.chain.from_iterable(
                itertools.combinations(inputs, r) for r in range(len(inputs) + 1)
            )
        ]

    return []

def joint_probability(people, one_gene, two_gene, trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_gene` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set `have_trait` does not have the trait.
    """

    joint_probability = float(1)
    for person in people:
        gene_id = (2 if person in two_gene else 1 if person in one_gene else 0)
        gene_probability = float(1)
        have_trait = person in trait
        father = people[person]['father']
        mother = people[person]['mother']
        parents = [father, mother]

        if father is None and mother is None:
            gene_probability = PROBS['gene'][gene_id]
        else:
            passing_probability = {}
            for parent in parents:
                if parent in two_gene:
                    passing_probability[parent] = 1 - PROBS['mutation']
                elif parent in one_gene:
                    passing_probability[parent] = 0.5
                else:
                    passing_probability[parent] = PROBS['mutation']

            if gene_id == 2:
                gene_probability = passing_probability[father]*passing_probability[mother]
            elif gene_id == 1:
                gene_probability = (1 - passing_probability[father]) * passing_probability[mother] + (1 - passing_probability[mother]) * passing_probability[father]
            else:
                gene_probability = (1 - passing_probability[father]) * (1 - passing_probability[mother])

        joint_probability = joint_probability * gene_probability
        trait_probability = PROBS['trait'][gene_id][have_trait]
        joint_probability = joint_probability * trait_probability

    return joint_probability

def update_probabilities(probabilities, one_gene_set, two_gene_set, trait_set, probability) -> None:
    """
    Update the probabilities dictionary with new probability values.

    :param probabilities: A dictionary of person to gene/trait probabilities
    :param one_gene_set: A set of people with one copy of the gene
    :param two_gene_set: A set of people with two copies of the gene
    :param trait_set: A set of people with the trait
    :param probability: The probability to add to the dictionary
    """
    for person, probabilities in probabilities.items():
        gene_id = 2 if person in two_gene_set else 1 if person in one_gene_set else 0
        have_trait = person in trait_set

        probabilities['gene'][gene_id] += probability
        probabilities['trait'][have_trait] += probability

def normalize_probabilities(probabilities_dict):
    """
    Normalize the probabilities dictionary by dividing each field's values by their sum.

    :param probabilities_dict: A dictionary of person to gene/trait probabilities
    """
    for person, fields in probabilities_dict.items():
        for field_name, values in fields.items():
            total = sum(values.values())
            for value, probability in values.items():
                values[value] = probability / total


def main():
    """
    Loads people data from a given CSV file, calculates the joint probability
    of each person having a particular gene and trait given the information
    from other people, and normalizes the resulting probabilities.

    Finally, it prints out the probabilities of each person having each gene
    and trait combination.

    :param sys.argv: A list containing the path to the CSV file
    :type sys.argv: list
    """
    if len(sys.argv) != 2:
        sys.exit('App usage: python app.py path_to_csv_file.csv')

    people = load_data(sys.argv[1])
    probabilities = {
        person: {
            'gene': {
                2: 0,
                1: 0,
                0: 0,
            },
            'trait': {
                True: 0,
                False: 0,
            }
        }
        for person in people
    }

    names = set(people)

    for have_trait in power_set(names):

        no_trait_evidence = any(
            people[person]['trait'] is not None and people[person]['trait'] != (person in have_trait)
            for person in people
        )

        if no_trait_evidence:
            continue

        for one_gene in power_set(names):
            for two_gene in power_set(names - one_gene):

                probability = joint_probability(people, one_gene, two_gene, have_trait)
                update_probabilities(probabilities, one_gene, two_gene, have_trait, probability)


        normalize_probabilities(probabilities)

    for person in people:
        print(f'{person}:')
        for field in probabilities[person]:
            print(f'  {field}:')
            for value in probabilities[person][field]:
                person_probability = probabilities[person][field][value]
                print(f'    {value}: {person_probability:.4f}')


if __name__ == '__main__':
    main()