#! /usr/bin/env bash

function abcli_image_classifier() {
    local task=$(abcli_unpack_keyword $1 help)

    if [ $task == "help" ] ; then
        abcli_help_line "$abcli_name image_classifier ingest [vegetable-images]" \
            "ingest vegetable-images for image_classifier."

        if [ "$(abcli_keyword_is $2 verbose)" == true ] ; then
            python3 -m image_classifier --help
        fi

        return
    fi

    if [ "$task" == "ingest" ] ; then
        local what=$(abcli_clarify_arg "$2" vegetable-images)

        if [ "$what" == "vegetable-images" ] ; then
            wget https://www.dropbox.com/s/lbqzfovdqs02nr8/vegetable-images.zip?dl=0
            unzip vegetable-images.zip
        else
            abcli_log_error "-image_classifier: ingest: $what: source not found."
        fi
        return
    fi

    abcli_log_error "-image_classifier: $task: command not found."
}