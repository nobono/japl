#ifndef _STAGED_H_
#define _STAGED_H_

#include <stdint.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <pybind11/stl.h>

using std::vector;
using std::string;


template<class T>
class Staged {

public:

    vector<T> stages;
    uint16_t stage_id = 0;
    bool is_stage = true;

    // Staged() {
    //     this->stage_id = 0;
    //     this->is_stage = true;
    // };

    Staged() = default;
    ~Staged() = default;

    // // copy constructor
    // Staged(const T& other)
    // :
    //     stages(other.stages),
    //     stage_id(other.stage_id),
    //     is_stage(other.is_stage) {}

    // T& operator=(const T& other) {
    //     if (this == &other) {
    //         return *this;
    //     }
    //     this->stages = other.stages;
    //     this->stage_id = other.stage_id;
    //     this->is_stage = other.is_stage;
    //     return *this;
    // }

    virtual void add_stage(T& stage) {
        this->is_stage = false;
        stage.is_stage = true;
        this->stages.push_back(stage);
    }

    virtual void set_stage(int& stage) {
        /*
         * Set the current stage index for the aerotable. This is
         * so that \"get_stage()\" will return the corresponding aerotable.
        */
        if (stage >= this->stages.size()) {
            string err_msg = "cannot access stage " + std::to_string(stage) +
                " for container of size " + std::to_string(this->stages.size());
            throw std::invalid_argument(err_msg);
        }
        this->stage_id = stage;
    }

    virtual T get_stage() const;
};


/* Trampoline class */
template<class T>
class PyStaged : public Staged<T> {
public:
    /* Inherit the constructors */
    using Staged<T>::Staged;

    /* Trampoline (need one for each virtual function) */
    void add_stage(T& stage) override {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            Staged<T>,   /* Parent class */
            add_stage,   /* Name of function in C++ (must match Python name) */
            stage        /* Argument(s) */
        );
    }

    void set_stage(int& stage) override {
        PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            Staged<T>,   /* Parent class */
            set_stage,   /* Name of function in C++ (must match Python name) */
            stage        /* Argument(s) */
        );
    }

    T get_stage() const override {
        PYBIND11_OVERRIDE_PURE(
            T, /* Return type */
            Staged<T>,   /* Parent class */
            get_stage,   /* Name of function in C++ (must match Python name) */
        );
    }

};



#endif
